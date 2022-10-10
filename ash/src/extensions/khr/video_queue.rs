use crate::prelude::*;
use crate::vk;
use crate::RawPtr;
use crate::{Device, Instance};
use std::ffi::CStr;
use std::mem;
use std::ptr;

/// <https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/VK_KHR_video_queue.html>
#[derive(Clone)]
pub struct VideoQueue {
    handle: vk::Device,
    fp: vk::KhrVideoQueueFn,
}

impl VideoQueue {
    pub fn new(instance: &Instance, device: &Device) -> Self {
        let handle = device.handle();
        let fp = vk::KhrVideoQueueFn::load(|name| unsafe {
            mem::transmute(instance.get_device_proc_addr(handle, name.as_ptr()))
        });
        Self { handle, fp }
    }

    /// <https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/vkCreateVideoSessionKHR.html>
    pub unsafe fn create_video_session(
        &self,
        create_info: &vk::VideoSessionCreateInfoKHR,
        allocation_callbacks: Option<&vk::AllocationCallbacks>,
    ) -> VkResult<vk::VideoSessionKHR> {
        let mut video_session = mem::zeroed();
        (self.fp.create_video_session_khr)(
            self.handle,
            create_info,
            allocation_callbacks.as_raw_ptr(),
            &mut video_session,
        )
        .result_with_success(video_session)
    }

    /// <https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/vkCreateVideoSessionParametersKHR.html>
    pub unsafe fn create_video_session_parameters(
        &self,
        create_info: &vk::VideoSessionParametersCreateInfoKHR,
        allocation_callbacks: Option<&vk::AllocationCallbacks>,
    ) -> VkResult<vk::VideoSessionParametersKHR> {
        let mut video_session_parameters = mem::zeroed();
        (self.fp.create_video_session_parameters_khr)(
            self.handle,
            create_info,
            allocation_callbacks.as_raw_ptr(),
            &mut video_session_parameters,
        )
        .result_with_success(video_session_parameters)
    }

    /// <https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/vkDestroyVideoSessionKHR.html>
    pub unsafe fn destroy_video_session(
        &self,
        video_session: vk::VideoSessionKHR,
        allocation_callbacks: Option<&vk::AllocationCallbacks>,
    ) {
        (self.fp.destroy_video_session_khr)(
            self.handle,
            video_session,
            allocation_callbacks.as_raw_ptr(),
        )
    }

    /// <https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/vkDestroyVideoSessionParametersKHR.html>
    pub unsafe fn destroy_video_session_parameters(
        &self,
        video_session_parameters: vk::VideoSessionParametersKHR,
        allocation_callbacks: Option<&vk::AllocationCallbacks>,
    ) {
        (self.fp.destroy_video_session_parameters_khr)(
            self.handle,
            video_session_parameters,
            allocation_callbacks.as_raw_ptr(),
        )
    }

    /// <https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/vkGetPhysicalDeviceVideoCapabilitiesKHR.html>
    pub unsafe fn get_physical_device_video_capabilities(
        &self,
        physical_device: vk::PhysicalDevice,
        video_profile: &vk::VideoProfileInfoKHR,
        capabilities: &mut vk::VideoCapabilitiesKHR,
    ) -> VkResult<()> {
        (self.fp.get_physical_device_video_capabilities_khr)(
            physical_device,
            video_profile,
            capabilities,
        )
        .result()
    }

    /// Retrieve the number of elements to pass to [`get_physical_device_video_format_properties()`][Self::get_physical_device_video_format_properties()]
    pub unsafe fn get_physical_device_video_format_properties_len(
        &self,
        physical_device: vk::PhysicalDevice,
        video_format_info: &vk::PhysicalDeviceVideoFormatInfoKHR,
    ) -> VkResult<usize> {
        let mut count = 0;
        (self.fp.get_physical_device_video_format_properties_khr)(
            physical_device,
            video_format_info,
            &mut count,
            ptr::null_mut(),
        )
        .result_with_success(count as usize)
    }

    /// <https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/vkGetPhysicalDeviceVideoFormatPropertiesKHR.html>
    ///
    /// Call [`get_physical_device_video_format_properties_len()`][Self::get_physical_device_video_format_properties_len()] to query the number of elements to pass to `out`.
    /// Be sure to [`Default::default()`]-initialize these elements and optionally set their `p_next` pointer.
    pub unsafe fn get_physical_device_video_format_properties(
        &self,
        physical_device: vk::PhysicalDevice,
        video_format_info: &vk::PhysicalDeviceVideoFormatInfoKHR,
        out: &mut [vk::VideoFormatPropertiesKHR],
    ) -> VkResult<()> {
        let mut count = out.len() as u32;
        (self.fp.get_physical_device_video_format_properties_khr)(
            physical_device,
            video_format_info,
            &mut count,
            out.as_mut_ptr(),
        )
        .result()?;
        assert_eq!(count as usize, out.len());
        Ok(())
    }

    /// Retrieve the number of elements to pass to [`get_video_session_memory_requirements()`][Self::get_video_session_memory_requirements()]
    pub unsafe fn get_video_session_memory_requirements_len(
        &self,
        video_session: vk::VideoSessionKHR,
    ) -> VkResult<usize> {
        let mut count = 0;
        (self.fp.get_video_session_memory_requirements_khr)(
            self.handle,
            video_session,
            &mut count,
            ptr::null_mut(),
        )
        .result_with_success(count as usize)
    }

    /// <https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/vkGetVideoSessionMemoryRequirementsKHR.html>
    ///
    /// Call [`get_video_session_memory_requirements_len()`][Self::get_video_session_memory_requirements_len()] to query the number of elements to pass to `out`.
    /// Be sure to [`Default::default()`]-initialize these elements and optionally set their `p_next` pointer.
    pub unsafe fn get_video_session_memory_requirements(
        &self,
        video_session: vk::VideoSessionKHR,
        out: &mut [vk::VideoSessionMemoryRequirementsKHR],
    ) -> VkResult<()> {
        let mut count = out.len() as u32;
        (self.fp.get_video_session_memory_requirements_khr)(
            self.handle,
            video_session,
            &mut count,
            out.as_mut_ptr(),
        )
        .result()?;
        assert_eq!(count as usize, out.len());
        Ok(())
    }

    /// <https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/vkUpdateVideoSessionParametersKHR.html>
    pub unsafe fn update_video_session_parameters(
        &self,
        video_session_parameters: vk::VideoSessionParametersKHR,
        update_info: &vk::VideoSessionParametersUpdateInfoKHR,
    ) -> VkResult<()> {
        (self.fp.update_video_session_parameters_khr)(
            self.handle,
            video_session_parameters,
            update_info,
        )
        .result()
    }

    /// <https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/vkBindVideoSessionMemoryKHR.html>
    pub unsafe fn bind_video_session_memory(
        &self,
        video_session: vk::VideoSessionKHR,
        video_session_bind_memories: &[vk::BindVideoSessionMemoryInfoKHR],
    ) -> VkResult<()> {
        (self.fp.bind_video_session_memory_khr)(
            self.handle,
            video_session,
            video_session_bind_memories.len() as u32,
            video_session_bind_memories.as_ptr(),
        )
        .result()
    }

    /// <https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/vkCmdBeginVideoCodingKHR.html>
    pub unsafe fn cmd_begin_video_coding(
        &self,
        command_buffer: vk::CommandBuffer,
        begin_info: &vk::VideoBeginCodingInfoKHR,
    ) {
        (self.fp.cmd_begin_video_coding_khr)(command_buffer, begin_info)
    }

    /// <https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/vkCmdControlVideoCodingKHR.html>
    pub unsafe fn cmd_control_video_coding(
        &self,
        command_buffer: vk::CommandBuffer,
        coding_control_info: &vk::VideoCodingControlInfoKHR,
    ) {
        (self.fp.cmd_control_video_coding_khr)(command_buffer, coding_control_info)
    }

    /// <https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/vkCmdEndVideoCodingKHR.html>
    pub unsafe fn cmd_end_video_coding(
        &self,
        command_buffer: vk::CommandBuffer,
        end_control_info: &vk::VideoEndCodingInfoKHR,
    ) {
        (self.fp.cmd_end_video_coding_khr)(command_buffer, end_control_info)
    }

    pub const fn name() -> &'static CStr {
        vk::KhrVideoQueueFn::name()
    }

    pub fn fp(&self) -> &vk::KhrVideoQueueFn {
        &self.fp
    }

    pub fn device(&self) -> vk::Device {
        self.handle
    }
}
