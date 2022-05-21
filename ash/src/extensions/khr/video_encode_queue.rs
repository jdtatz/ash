use crate::vk;
use crate::{Device, Instance};
use std::ffi::CStr;
use std::mem;

/// <https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/VK_KHR_video_encode_queue.html>
#[derive(Clone)]
pub struct VideoEncodeQueue {
    handle: vk::Device,
    fp: vk::KhrVideoEncodeQueueFn,
}

impl VideoEncodeQueue {
    pub fn new(instance: &Instance, device: &Device) -> Self {
        let handle = device.handle();
        let fp = vk::KhrVideoEncodeQueueFn::load(|name| unsafe {
            mem::transmute(instance.get_device_proc_addr(handle, name.as_ptr()))
        });
        Self { handle, fp }
    }

    /// <https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/vkCmdEncodeVideoKHR.html>
    pub unsafe fn cmd_encode_video(
        &self,
        command_buffer: vk::CommandBuffer,
        encode_info: &vk::VideoEncodeInfoKHR,
    ) {
        (self.fp.cmd_encode_video_khr)(command_buffer, encode_info)
    }

    pub const fn name() -> &'static CStr {
        vk::KhrVideoEncodeQueueFn::name()
    }

    pub fn fp(&self) -> &vk::KhrVideoEncodeQueueFn {
        &self.fp
    }

    pub fn device(&self) -> vk::Device {
        self.handle
    }
}
