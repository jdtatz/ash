use std::ptr::{addr_of, addr_of_mut};

use crate::vk;

/// Holds 24 bits in the least significant bits of memory,
/// and 8 bytes in the most significant bits of that memory,
/// occupying a single [`u32`] in total. This is commonly used in
/// [acceleration structure instances] such as
/// [`vk::AccelerationStructureInstanceKHR`],
/// [`vk::AccelerationStructureSRTMotionInstanceNV`] and
/// [`vk::AccelerationStructureMatrixMotionInstanceNV`].
///
/// [acceleration structure instances]: https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/VkAccelerationStructureInstanceKHR.html#_description
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
#[repr(transparent)]
pub struct Packed24_8(u32);

impl Packed24_8 {
    pub fn new(low_24: u32, high_8: u8) -> Self {
        Self((low_24 & 0x00ff_ffff) | (u32::from(high_8) << 24))
    }

    /// Extracts the least-significant 24 bits (3 bytes) of this integer
    pub fn low_24(&self) -> u32 {
        self.0 & 0xffffff
    }

    /// Extracts the most significant 8 bits (single byte) of this integer
    pub fn high_8(&self) -> u8 {
        (self.0 >> 24) as u8
    }
}

// Intradoc `Self::` links refuse to resolve if `ColorComponentFlags`
// isn't directly in scope: https://github.com/rust-lang/rust/issues/93205
use vk::ColorComponentFlags;

impl ColorComponentFlags {
    /// Contraction of [`R`][Self::R] | [`G`][Self::G] | [`B`][Self::B] | [`A`][Self::A]
    pub const RGBA: Self = Self(Self::R.0 | Self::G.0 | Self::B.0 | Self::A.0);
}

impl From<vk::Extent2D> for vk::Extent3D {
    fn from(value: vk::Extent2D) -> Self {
        Self {
            width: value.width,
            height: value.height,
            depth: 1,
        }
    }
}

impl From<vk::Extent2D> for vk::Rect2D {
    fn from(extent: vk::Extent2D) -> Self {
        Self {
            offset: Default::default(),
            extent,
        }
    }
}

pub unsafe trait Extends<Base: ?Sized> {}

/// Structures implementing this trait are layout-compatible with [`vk::BaseInStructure`] and
/// [`vk::BaseOutStructure`]. Such structures have an `s_type` field indicating its type, which
/// must always match the value of [`TaggedStructure::STRUCTURE_TYPE`].
pub unsafe trait TaggedStructure {
    const STRUCTURE_TYPE: vk::StructureType;

    /// Prepends the given extension struct between the root and the first pointer. This
    /// method only exists on structs that can be passed to a function directly. Only
    /// valid extension structs can be pushed into the chain.
    /// If the chain looks like `A -> B -> C`, and you call `x.push_next(&mut D)`, then the
    /// chain will look like `A -> D -> B -> C`.
    fn push_next<T: TaggedStructure + Extends<Self>>(mut self, next: &mut T) -> Self
    where
        Self: Sized,
    {
        unsafe {
            // `next` here can contain a pointer chain. This means that we must correctly
            // attach he head to the root and the tail to the rest of the chain
            // For example:
            //
            // next = A -> B
            // Before: `Root -> C -> D -> E`
            // After: `Root -> A -> B -> C -> D -> E`
            //                 ^^^^^^
            //                 next chain
            let last_next = vk::ptr_chain_iter(next).last().unwrap();
            (*last_next).p_next = (*(addr_of!(self) as *const vk::BaseOutStructure)).p_next;
            (*(addr_of_mut!(self) as *mut vk::BaseOutStructure)).p_next = <*mut T>::cast(next) as *mut vk::BaseOutStructure;
        }
        self
    }
}
