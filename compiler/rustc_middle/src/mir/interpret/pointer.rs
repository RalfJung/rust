use super::{AllocId, InterpResult};

use rustc_macros::HashStable;
use rustc_target::abi::{HasDataLayout, Size};

use std::convert::TryFrom;
use std::fmt;

////////////////////////////////////////////////////////////////////////////////
// Pointer arithmetic
////////////////////////////////////////////////////////////////////////////////

pub trait PointerArithmetic: HasDataLayout {
    // These are not supposed to be overridden.

    #[inline(always)]
    fn pointer_size(&self) -> Size {
        self.data_layout().pointer_size
    }

    #[inline]
    fn machine_usize_max(&self) -> u64 {
        let max_usize_plus_1 = 1u128 << self.pointer_size().bits();
        u64::try_from(max_usize_plus_1 - 1).unwrap()
    }

    #[inline]
    fn machine_isize_min(&self) -> i64 {
        let max_isize_plus_1 = 1i128 << (self.pointer_size().bits() - 1);
        i64::try_from(-max_isize_plus_1).unwrap()
    }

    #[inline]
    fn machine_isize_max(&self) -> i64 {
        let max_isize_plus_1 = 1u128 << (self.pointer_size().bits() - 1);
        i64::try_from(max_isize_plus_1 - 1).unwrap()
    }

    /// Helper function: truncate given value-"overflowed flag" pair to pointer size and
    /// update "overflowed flag" if there was an overflow.
    /// This should be called by all the other methods before returning!
    #[inline]
    fn truncate_to_ptr(&self, (val, over): (u64, bool)) -> (u64, bool) {
        let val = u128::from(val);
        let max_ptr_plus_1 = 1u128 << self.pointer_size().bits();
        (u64::try_from(val % max_ptr_plus_1).unwrap(), over || val >= max_ptr_plus_1)
    }

    #[inline]
    fn overflowing_offset(&self, val: u64, i: u64) -> (u64, bool) {
        // We do not need to check if i fits in a machine usize. If it doesn't,
        // either the wrapping_add will wrap or res will not fit in a pointer.
        let res = val.overflowing_add(i);
        self.truncate_to_ptr(res)
    }

    #[inline]
    fn overflowing_signed_offset(&self, val: u64, i: i64) -> (u64, bool) {
        // We need to make sure that i fits in a machine isize.
        let n = i.unsigned_abs();
        if i >= 0 {
            let (val, over) = self.overflowing_offset(val, n);
            (val, over || i > self.machine_isize_max())
        } else {
            let res = val.overflowing_sub(n);
            let (val, over) = self.truncate_to_ptr(res);
            (val, over || i < self.machine_isize_min())
        }
    }

    #[inline]
    fn offset<'tcx>(&self, val: u64, i: u64) -> InterpResult<'tcx, u64> {
        let (res, over) = self.overflowing_offset(val, i);
        if over { throw_ub!(PointerArithOverflow) } else { Ok(res) }
    }

    #[inline]
    fn signed_offset<'tcx>(&self, val: u64, i: i64) -> InterpResult<'tcx, u64> {
        let (res, over) = self.overflowing_signed_offset(val, i);
        if over { throw_ub!(PointerArithOverflow) } else { Ok(res) }
    }
}

impl<T: HasDataLayout> PointerArithmetic for T {}

/// This trait abstracts over the kind of provenance that is associated with a `Pointer`. It is
/// mostly opaque; the `Machine` trait extends it with some more operations that also have access to
/// some global state.
pub trait Provenance: Copy {
    /// Says whether the `offset` field of `Pointer`s with this provenance is the actual physical address.
    /// If `true, ptr-to-int casts work by simply discarding the provenance.
    /// If `false`, ptr-to-int casts are not supported.
    fn offset_is_addr(self) -> bool;

    /// Determines how a pointer should be printed.
    fn fmt(ptr: &Pointer<Self>, f: &mut fmt::Formatter<'_>) -> fmt::Result
    where
        Self: Sized;

    /// "Erasing" a tag converts it to the default tag type. Used only for formatting purposes!
    fn erase_for_fmt(self) -> DefaultTag;
}

/// The default provenance "tag" used by compile-time function evaluation.
/// Needs to be an `Option` to implement `Machine::provenance_from_addr`;
/// pointers tagged with `None` are absolute integer addresses.
pub type DefaultTag = Option<AllocId>;

// This impl is used for `ConstValue` -- the result of CTFE queries avoids possible `None`
// provenances.
impl Provenance for AllocId {
    // With the `AllocId` as provenance, the `offset` is interpreted *relative to the allocation*,
    // so ptr-to-int casts are not possible (since we do not know the global physical offset).
    #[inline(always)]
    fn offset_is_addr(self) -> bool {
        false
    }

    fn fmt(ptr: &Pointer<Self>, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Forward `alternate` flag to `alloc_id` printing.
        if f.alternate() {
            write!(f, "{:#?}", ptr.provenance)?;
        } else {
            write!(f, "{:?}", ptr.provenance)?;
        }
        // Print offset only if it is non-zero.
        if ptr.offset.bytes() > 0 {
            write!(f, "+0x{:x}", ptr.offset.bytes())?;
        }
        Ok(())
    }

    fn erase_for_fmt(self) -> DefaultTag {
        Some(self)
    }
}

impl Provenance for DefaultTag {
    // With the `Option<AllocId>` as provenance, the `offset` is interpreted *relative to the
    // allocation*, so ptr-to-int casts are not possible (since we do not know the global physical
    // offset) -- except if the provenance is `None`, which indicates regular integers.
    #[inline(always)]
    fn offset_is_addr(self) -> bool {
        self.is_none()
    }

    fn fmt(ptr: &Pointer<Self>, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match ptr.provenance {
            Some(alloc_id) => {
                Provenance::fmt(&Pointer { offset: ptr.offset, provenance: alloc_id }, f)
            }
            None => {
                // An integer as a pointer.
                write!(f, "0x{:x}", ptr.offset.bytes())
            }
        }
    }

    fn erase_for_fmt(self) -> DefaultTag {
        self
    }
}

/// Represents a pointer in the Miri engine.
///
/// Pointers are "tagged" with provenance information; typically the `AllocId` they belong to.
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, TyEncodable, TyDecodable, Hash)]
#[derive(HashStable)]
pub struct Pointer<Tag = DefaultTag> {
    pub(super) offset: Size, // kept private to avoid accidental misinterpretation (meaning depends on `Tag` type)
    pub provenance: Tag,
}

//FIXME static_assert_size!(Pointer, 16);

// We want the `Debug` output to be readable as it is used by `derive(Debug)` for
// all the Miri types.
impl<Tag: Provenance> fmt::Debug for Pointer<Tag> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Tag::fmt(self, f)
    }
}

/// Produces a `Pointer` that points to the beginning of the `Allocation`.
impl From<AllocId> for Pointer {
    #[inline(always)]
    fn from(alloc_id: AllocId) -> Self {
        Pointer::new(Some(alloc_id), Size::ZERO)
    }
}
impl From<AllocId> for Pointer<AllocId> {
    #[inline(always)]
    fn from(alloc_id: AllocId) -> Self {
        Pointer::new(alloc_id, Size::ZERO)
    }
}
impl From<Pointer<AllocId>> for Pointer {
    #[inline(always)]
    fn from(ptr: Pointer<AllocId>) -> Self {
        Pointer::new(Some(ptr.provenance), ptr.offset)
    }
}
impl TryFrom<Pointer> for Pointer<AllocId> {
    type Error = ();

    #[inline(always)]
    fn try_from(ptr: Pointer) -> Result<Self, ()> {
        if let Some(alloc_id) = ptr.provenance {
            Ok(Pointer::new(alloc_id, ptr.offset))
        } else {
            Err(())
        }
    }
}

impl<'tcx, Tag> Pointer<Tag> {
    #[inline(always)]
    pub fn new(provenance: Tag, offset: Size) -> Self {
        Pointer { provenance, offset }
    }

    /// Obtain the constituents of this pointer. Not that the meaning of the offset depends on the type `Tag`!
    /// This function must only be used in the implementation of `Machine::ptr_get_alloc`,
    /// and when a `Pointer` is taken apart to be stored efficiently in an `Allocation`.
    #[inline(always)]
    pub fn into_parts(self) -> (Tag, Size) {
        (self.provenance, self.offset)
    }

    #[inline(always)]
    pub fn erase_for_fmt(self) -> Pointer
    where
        Tag: Provenance,
    {
        Pointer { offset: self.offset, provenance: self.provenance.erase_for_fmt() }
    }

    #[inline]
    pub fn offset(self, i: Size, cx: &impl HasDataLayout) -> InterpResult<'tcx, Self> {
        Ok(Pointer {
            offset: Size::from_bytes(cx.data_layout().offset(self.offset.bytes(), i.bytes())?),
            ..self
        })
    }

    #[inline]
    pub fn overflowing_offset(self, i: Size, cx: &impl HasDataLayout) -> (Self, bool) {
        let (res, over) = cx.data_layout().overflowing_offset(self.offset.bytes(), i.bytes());
        let ptr = Pointer { offset: Size::from_bytes(res), ..self };
        (ptr, over)
    }

    #[inline(always)]
    pub fn wrapping_offset(self, i: Size, cx: &impl HasDataLayout) -> Self {
        self.overflowing_offset(i, cx).0
    }

    #[inline]
    pub fn signed_offset(self, i: i64, cx: &impl HasDataLayout) -> InterpResult<'tcx, Self> {
        Ok(Pointer {
            offset: Size::from_bytes(cx.data_layout().signed_offset(self.offset.bytes(), i)?),
            ..self
        })
    }

    #[inline]
    pub fn overflowing_signed_offset(self, i: i64, cx: &impl HasDataLayout) -> (Self, bool) {
        let (res, over) = cx.data_layout().overflowing_signed_offset(self.offset.bytes(), i);
        let ptr = Pointer { offset: Size::from_bytes(res), ..self };
        (ptr, over)
    }

    #[inline(always)]
    pub fn wrapping_signed_offset(self, i: i64, cx: &impl HasDataLayout) -> Self {
        self.overflowing_signed_offset(i, cx).0
    }
}
