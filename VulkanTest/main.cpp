#define GLFW_INCLUDE_VULKAN //see Line 110 of glfw3.h
#include <GLFW/glfw3.h>

#include <iostream>
#include <stdexcept>
#include <cstdlib>

#include<optional> //for graphicsFamily queue suitability

#include<vector> 

#include<map> //for ranking GPU suitability (and perhaps other things) 

#include<set> //has to do with presentQueue

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

const std::vector<const char*> validationLayers =
{
	"VK_LAYER_KHRONOS_validation"
};

const std::vector<const char*> deviceExtensions = {
	VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

#ifdef NDEBUG
const bool enableValidationLayers = false; 
#else
const bool enableValidationLayers = true;
#endif // NDEBUG


VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, 
	const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, 
	const VkAllocationCallbacks* pAllocator, 
	VkDebugUtilsMessengerEXT* pDebugMessenger)
{
	auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
	if (func != nullptr) {
		return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
	}
	else {
		return VK_ERROR_EXTENSION_NOT_PRESENT;
	}
}


void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
	auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
	if (func != nullptr) {
		func(instance, debugMessenger, pAllocator);
	}
}


struct QueueFamilyIndices
{
	std::optional<uint32_t> graphicsFamily; 
	std::optional<uint32_t> presentFamily; 

	bool isComplete() {
		return graphicsFamily.has_value() && presentFamily.has_value(); 
	}
};



class HelloTriangleApplication {
public:
	void run()
	{
		initWindow(); 
		initVulkan(); 
		mainLoop(); 
		cleanup(); 
	}

private:
	GLFWwindow* window;
	VkInstance instance; 
	VkDebugUtilsMessengerEXT debugMessenger; 
	VkSurfaceKHR surface;

	VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
	VkDevice device; //careful not to mix this one up -> this is the LOGICAL device (not physical)
	
	VkQueue graphicsQueue;
	VkQueue presentQueue; //as in "presentation" queue (NOT as in the currentQueue)

	bool checkValidationLayerSupport()
	{
		uint32_t layerCount; 
		vkEnumerateInstanceLayerProperties(&layerCount, nullptr); 

		std::vector<VkLayerProperties> availableLayers(layerCount);
		vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

		//return false;

		for (const char* layerName : validationLayers)
		{
			bool layerFound = false;

			for (const auto& layerProperties : availableLayers)
			{
				if (strcmp(layerName, layerProperties.layerName) == 0)
				{
					layerFound = true;
					break;
				}
			}
			if (!layerFound)
			{
				return false;
			}
		
		}
		return true; 
	}

	std::vector<const char*> getRequiredExtensions()
	{
		uint32_t glfwExtensionCount = 0; 
		const char** glfwExtensions; 
		glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount); 

		std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

		if (enableValidationLayers)
		{
			extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME); 
		}
		return extensions; 

	}

	static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
		VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
		VkDebugUtilsMessageTypeFlagsEXT messageType,
		const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
		void* pUserData)
	{
		std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl; 

		return VK_FALSE; //0U ...
	}

	//the "important" functions: 
	void initWindow()
	{
		glfwInit();
		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); 
		glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE); 

		window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
	}

	void createInstance()
	{
		if (enableValidationLayers && !checkValidationLayerSupport())
		{
			throw std::runtime_error("validation layers requested but not available");
		}

		auto extensions = getRequiredExtensions(); 
		
		//the appInfo struct object is optional 
		VkApplicationInfo appInfo{}; 
		appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO; //has value 0
		appInfo.pApplicationName = "Hello Triangle";
		appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0); //deprecated warning/suggestion
		appInfo.pEngineName = "No Engine"; 
		appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0); 
		appInfo.apiVersion = VK_API_VERSION_1_0; 

		//creatInfo struct is MANDATORY
		VkInstanceCreateInfo createInfo{}; 
		createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO; //has value 1
		createInfo.pApplicationInfo = &appInfo; 
		
		//uint32_t glfwExtensionCount = 0; 
		//const char** glfwExtensions; 

		//glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount); 

		createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
		createInfo.ppEnabledExtensionNames = extensions.data();

		VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};

		if (enableValidationLayers)
		{
			createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
			createInfo.ppEnabledLayerNames = validationLayers.data(); 

			populateDebugMessengerCreateInfo(debugCreateInfo); 

			createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;

		}
		else
		{
			createInfo.enabledLayerCount = 0; 
			createInfo.pNext = nullptr; 
		}

		/*----------End of `createInfo` struct initialization--------------*/

		VkResult result = vkCreateInstance(&createInfo, nullptr, &instance);

		if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) //VK_SUCCESS = 0 (not 1) 
		{
			throw std::runtime_error("failed to create instance");
		}
	}

	void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
		createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
		createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
		createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
		createInfo.pfnUserCallback = debugCallback;
	}

	void setupDebugMessenger()
	{
		if (!enableValidationLayers)
		{
			return; 
		}

		VkDebugUtilsMessengerCreateInfoEXT createInfo;
		populateDebugMessengerCreateInfo(createInfo);

		if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
			throw std::runtime_error("failed to set up debug messenger!");
		}
	}

	void createSurface()
	{
		if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
			throw std::runtime_error("failed to create window surface!");
		}
	}

	//simple version - used by `pickPhysicalDevice`
	bool isDeviceSuitable(VkPhysicalDevice device)
	{
		QueueFamilyIndices indices = findQueueFamilies(device); 

		bool extensionsSupported = checkDeviceExtensionSupport(device);

		//std::cout << "\n\nExtentsion supported? " << extensionsSupported << std::endl; 
		return indices.isComplete() && extensionsSupported; 
	}

	bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
		uint32_t extensionCount;
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

		std::vector<VkExtensionProperties> availableExtensions(extensionCount);
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

		std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

		for (const auto& extension : availableExtensions) {
			requiredExtensions.erase(extension.extensionName);
		}

		return requiredExtensions.empty();
	}

	QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device)
	{
		QueueFamilyIndices indices;

		uint32_t queueFamilyCount = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

		std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

		int i = 0;
		for (const auto& queueFamily : queueFamilies) {
			if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) //BITWISE and!
			{
				indices.graphicsFamily = i;
			}


			VkBool32 presentSupport = false;
			vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);

			if (presentSupport) {
				indices.presentFamily = i;
			}

			if (indices.isComplete())
			{
				break;
			}

			i++; //toggling-type code, I think
		}
		return indices;
	}

	//complicated version - rating system (favors NVidia graphics card on my computer) 
	int rateDeviceSuitability(VkPhysicalDevice device)
	{
		VkPhysicalDeviceProperties deviceProperties;
		VkPhysicalDeviceFeatures deviceFeatures;

		vkGetPhysicalDeviceProperties(device, &deviceProperties);
		vkGetPhysicalDeviceFeatures(device, &deviceFeatures);

		int score = 0; 

		if (deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
			score += 1000; //arbitrary 1000, I suppose
		}

		// Maximum possible size of textures affects graphics quality
		score += deviceProperties.limits.maxImageDimension2D;

		// Application can't function without geometry shaders
		if (!deviceFeatures.geometryShader) {
			return 0;
		}

		return score; 
	}

	void pickPhysicalDevice()
	{
		uint32_t deviceCount = 0; 
		vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr); 

		if (deviceCount == 0)
		{
			throw std::runtime_error("failed to find GPUs with Vulkan support");
		}

		std::vector<VkPhysicalDevice> devices(deviceCount);
		vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());


		//UNCOMMENT if using `isDeviceSuitable` (will select integrated graphics on my computer)
		
		for (const auto& device : devices)
		{
			if (isDeviceSuitable(device))
			{
				physicalDevice = device; 
				break; 
			}
		}

		if (physicalDevice == VK_NULL_HANDLE)
		{
			throw std::runtime_error("Could not find suitable device");
		}

		//else //optional print 
		//{
		//	VkPhysicalDeviceProperties deviceProperties;
		//	vkGetPhysicalDeviceProperties(physicalDevice, &deviceProperties);
		//	std::cout << "Selected this device: " << deviceProperties.deviceName << std::endl; 
		//}

		
		//uncomment if using `rateDeviceSuitability` (will pick NVIDIA graphics card)
		
		/*std::multimap<int, VkPhysicalDevice> candidates;
		
		for (const auto& device : devices)
		{
			int score = rateDeviceSuitability(device);

			candidates.insert(std::make_pair(score, device)); 
		}

		if (candidates.rbegin()->first > 0) //rbegin -> map is ordered by key (score) 
		{
			physicalDevice = candidates.rbegin()->second; 

			//optional print: 
			//VkPhysicalDeviceProperties deviceProperties;
			//vkGetPhysicalDeviceProperties(physicalDevice, &deviceProperties);
			//std::cout << "Selected this device: " << deviceProperties.deviceName << std::endl; 
		}
		else
		{
			throw std::runtime_error("failed to find a SUITABLE GPU");
		}
		*/
	}


	void createLogicalDevice()
	{
		QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

		std::vector<VkDeviceQueueCreateInfo> queueCreateInfos; //plural now - has to do with presentQueue
		
		std::set<uint32_t> uniqueQueueFamilies = { indices.graphicsFamily.value(),
			indices.presentFamily.value() };


		float queuePriority = 1.0f;
		
		for (uint32_t queueFamily : uniqueQueueFamilies)
		{
			VkDeviceQueueCreateInfo queueCreateInfo{};
			queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
			queueCreateInfo.queueFamilyIndex = queueFamily; 
			queueCreateInfo.queueCount = 1; //only 1 needed due to threading ability
			queueCreateInfo.pQueuePriorities = &queuePriority;

			queueCreateInfos.push_back(queueCreateInfo);

		}

		VkPhysicalDeviceFeatures deviceFeatures{};//all members set to false, for now

		VkDeviceCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
	
		createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
		createInfo.pQueueCreateInfos = queueCreateInfos.data();

		createInfo.pEnabledFeatures = &deviceFeatures;

		createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
		createInfo.ppEnabledExtensionNames = deviceExtensions.data(); 

		if (enableValidationLayers) {
			createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
			createInfo.ppEnabledLayerNames = validationLayers.data();
		}
		else {
			createInfo.enabledLayerCount = 0;
		}


		if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS)
		{
			throw std::runtime_error("could not create logical device!");
		}

		vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
		vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
	}

	void initVulkan() {
		createInstance(); 
		setupDebugMessenger(); 
		createSurface(); 
		pickPhysicalDevice(); 
		createLogicalDevice(); 
	}

	void mainLoop() {
		while (!glfwWindowShouldClose(window))
		{
			glfwPollEvents(); 
		}
	}

	void cleanup() {

		vkDestroyDevice(device, nullptr);

		if (enableValidationLayers)
		{
			DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr); 
		}


		vkDestroySurfaceKHR(instance, surface, nullptr);
		vkDestroyInstance(instance, nullptr);
		
		glfwDestroyWindow(window); 
		glfwTerminate(); 

	}
};

int main()
{
	HelloTriangleApplication app; 

	try
	{
		app.run(); 
	}

	catch (const std::exception& e)
	{
		std::cerr << e.what() << std::endl; 
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS; 
}