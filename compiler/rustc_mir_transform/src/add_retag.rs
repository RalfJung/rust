//! This pass adds validation calls (AcquireValid, ReleaseValid) where appropriate.
//! It has to be run really early, before transformations like inlining, because
//! introducing these calls *adds* UB -- so, conceptually, this pass is actually part
//! of MIR building, and only after this pass we think of the program has having the
//! normal MIR semantics.

use rustc_middle::mir::*;
use rustc_middle::ty::{self, Ty, TyCtxt};

pub struct AddRetag;

/// Determine whether this type may contain a reference (or box), and thus needs retagging.
/// We will only recurse `depth` times into Tuples/ADTs to bound the cost of this.
fn may_contain_reference<'tcx>(ty: Ty<'tcx>, depth: u32, tcx: TyCtxt<'tcx>) -> bool {
    match ty.kind() {
        // Primitive types that are not references
        ty::Bool
        | ty::Char
        | ty::Float(_)
        | ty::Int(_)
        | ty::Uint(_)
        | ty::RawPtr(..)
        | ty::FnPtr(..)
        | ty::Str
        | ty::FnDef(..)
        | ty::Never => false,
        // References
        ty::Ref(..) => true,
        ty::Adt(..) if ty.is_box() => true,
        ty::Adt(adt, _) if Some(adt.did()) == tcx.lang_items().ptr_unique() => true,
        // Compound types: recurse
        ty::Array(ty, _) | ty::Slice(ty) => {
            // This does not branch so we keep the depth the same.
            may_contain_reference(*ty, depth, tcx)
        }
        ty::Tuple(tys) => {
            depth == 0 || tys.iter().any(|ty| may_contain_reference(ty, depth - 1, tcx))
        }
        ty::Adt(adt, args) => {
            depth == 0
                || adt.variants().iter().any(|v| {
                    v.fields.iter().any(|f| may_contain_reference(f.ty(tcx, args), depth - 1, tcx))
                })
        }
        // Conservative fallback
        _ => true,
    }
}

impl<'tcx> MirPass<'tcx> for AddRetag {
    fn is_enabled(&self, sess: &rustc_session::Session) -> bool {
        sess.opts.unstable_opts.mir_emit_retag
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        // We need an `AllCallEdges` pass before we can do any work.
        super::add_call_guards::AllCallEdges.run_pass(tcx, body);

        let basic_blocks = body.basic_blocks.as_mut();
        let local_decls = &body.local_decls;
        let needs_retag = |place: &Place<'tcx>| {
            !place.is_indirect_first_projection() // we're not really interested in stores to "outside" locations, they are hard to keep track of anyway
                && may_contain_reference(place.ty(&*local_decls, tcx).ty, /*depth*/ 3, tcx)
                && !local_decls[place.local].is_deref_temp()
        };

        // PART 1
        // Retag arguments at the beginning of the start block.
        {
            // Gather all arguments, skip return value.
            let places = local_decls.iter_enumerated().skip(1).take(body.arg_count).filter_map(
                |(local, decl)| {
                    let place = Place::from(local);
                    needs_retag(&place).then_some((place, decl.source_info))
                },
            );

            // Emit their retags.
            basic_blocks[START_BLOCK].statements.splice(
                0..0,
                places.map(|(place, source_info)| Statement {
                    source_info,
                    kind: StatementKind::Retag(RetagKind::FnEntry, Box::new(place)),
                }),
            );
        }

        // PART 2
        // Retag return values of functions.
        // We collect the return destinations because we cannot mutate while iterating.
        let returns = basic_blocks
            .iter_mut()
            .filter_map(|block_data| {
                match block_data.terminator().kind {
                    TerminatorKind::Call { target: Some(target), destination, .. }
                        if needs_retag(&destination) =>
                    {
                        // Remember the return destination for later
                        Some((block_data.terminator().source_info, destination, target))
                    }

                    // `Drop` is also a call, but it doesn't return anything so we are good.
                    TerminatorKind::Drop { .. } => None,
                    // Not a block ending in a Call -> ignore.
                    _ => None,
                }
            })
            .collect::<Vec<_>>();
        // Now we go over the returns we collected to retag the return values.
        for (source_info, dest_place, dest_block) in returns {
            basic_blocks[dest_block].statements.insert(
                0,
                Statement {
                    source_info,
                    kind: StatementKind::Retag(RetagKind::Default, Box::new(dest_place)),
                },
            );
        }

        // PART 3
        // Add retag after assignments.
        for block_data in basic_blocks.iter_mut() {
            // We want to insert statements as we iterate. To this end, we
            // iterate backwards using indices.
            for i in (0..block_data.statements.len()).rev() {
                let (retag_kind, place) = match block_data.statements[i].kind {
                    // Retag after assignments of reference type.
                    StatementKind::Assign(box (ref place, ref rvalue)) if needs_retag(place) => {
                        let add_retag = match rvalue {
                            // Ptr-creating operations already do their own internal retagging, no
                            // need to also add a retag statement.
                            Rvalue::Ref(..) | Rvalue::AddressOf(..) => false,
                            _ => true,
                        };
                        if add_retag {
                            (RetagKind::Default, *place)
                        } else {
                            continue;
                        }
                    }
                    // Do nothing for the rest
                    _ => continue,
                };
                // Insert a retag after the statement.
                let source_info = block_data.statements[i].source_info;
                block_data.statements.insert(
                    i + 1,
                    Statement {
                        source_info,
                        kind: StatementKind::Retag(retag_kind, Box::new(place)),
                    },
                );
            }
        }

        // PART 4: special hack for calling `slice::len`, to avoid bad interactions with Stacked Borrows.
        // We basically perform lower_slice_len here, except we also remove the original reference
        // that was created for the function argument.
        let slice_len_fn = tcx.lang_items().slice_len_fn();
        for block_data in basic_blocks.iter_mut() {
            if let Some(Terminator {
                kind: TerminatorKind::Call { func, args, destination, target, .. },
                source_info,
            }) = &block_data.terminator
            {
                let func_ty = func.ty(&*local_decls, tcx);
                if let ty::FnDef(def_id, ..) = *func_ty.kind() {
                    if Some(def_id) == slice_len_fn && args.len() == 1 {
                        // Let's see if the last statement of the basic block is the argument to this function.
                        // This detetcs the pattern
                        // ```
                        // _arg = &*orig;
                        // Call(_arg)
                        // ```
                        if let Some(arg) = args[0].node.place().and_then(|p| p.as_local())
                            && let Some(last_stmt) = block_data.statements.last()
                            // Last statement must be an assignment where the LHS is the argument local.
                            && let StatementKind::Assign(box (ref arg_place, ref rvalue)) = last_stmt.kind
                            && arg_place.as_local() == Some(arg)
                            // RHS of that assignment must be a borrow.
                            && let Rvalue::Ref(_, _, orig_place) = *rvalue
                        {
                            // We have to remove the old assignment to make sure the retag does
                            // not happen. Let's make sure it's not something user-defined.
                            // We then rely on this local being used here and only here...
                            assert!(!matches!(local_decls[arg].local_info(), LocalInfo::User(..)));
                            block_data.statements.pop();
                            // Insert assignment that does what the function call did.
                            block_data.statements.push(Statement {
                                kind: StatementKind::Assign(Box::new((
                                    *destination,
                                    Rvalue::Len(orig_place),
                                ))),
                                source_info: *source_info,
                            });
                            // Replace call by jump to next basic block.
                            block_data.terminator_mut().kind =
                                TerminatorKind::Goto { target: target.unwrap() };
                        }
                    }
                }
            }
        }
    }
}
