// SPDX-FileCopyrightText: 2021-2026 Roy Shilkrot <roy.shil@gmail.com>
// SPDX-FileCopyrightText: 2023-2026 Kaito Udagawa <umireon@kaito.tokyo>
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "background-filter.h"

#include <onnxruntime_cxx_api.h>

#include <opencv2/core/version.hpp>

#if CV_VERSION_MAJOR >= 5
#include <opencv2/geometry.hpp>
#endif

#include <opencv2/imgproc.hpp>

#include <memory>
#include <mutex>
#include <regex>
#include <thread>
#include <condition_variable>
#include <chrono>

#include <util/platform.h>

#include <mach/mach.h>

#include "plugin-support.h"
#include "models/ModelMediapipe.hpp"
#include "models/ModelRVM.hpp"
#include "FilterData.hpp"
#include "ort-utils/ort-session-utils.hpp"
#include "obs-utils/obs-utils.hpp"
#include "consts.h"

struct background_removal_filter : public filter_data, public std::enable_shared_from_this<background_removal_filter> {
	bool stopWhenSourceIsInactive = true;
	float threshold = 0.5f;
	int maskExpansion = 0;
	int maskEveryXFrames = 1;
	int maskEveryXFramesCount = 0;
	int64_t blurBackground = 0;

	// Robust Video Matting resolution (0 fast, 1 balanced, 2 high)
	int rvmQuality = 1;

	cv::Mat backgroundMask;   // 255 = background, at frame size; read by the render thread under outputLock
	cv::Mat lastPersonAlpha;  // previous person probability, for temporal smoothing

	// Memory watchdog (see checkMemoryWatchdog)
	uint64_t memBaselineBytes = 0;
	uint64_t memWatchStartNs = 0;
	uint64_t memLastCheckNs = 0;
	uint64_t memLastActionNs = 0;
	int memStrikes = 0;
	std::atomic<bool> memoryStopped{false};

	// Mask worker thread: inference runs here, never on OBS's graphics thread
	std::thread worker;
	std::mutex workMutex;
	std::condition_variable workCv;
	bool workPending = false;
	bool workerStop = false;

	gs_effect_t *effect = nullptr;
	gs_effect_t *kawaseBlurEffect = nullptr;

	std::mutex modelMutex;

	~background_removal_filter() { obs_log(LOG_INFO, "Background removal filter destructor called"); }
};

const char *background_filter_getname(void *unused)
{
	UNUSED_PARAMETER(unused);
	return obs_module_text("BackgroundRemoval");
}

/**                   PROPERTIES                     */

static bool enable_advanced_settings(obs_properties_t *ppts, obs_property_t *p, obs_data_t *settings)
{
	UNUSED_PARAMETER(p);
	const bool enabled = obs_data_get_bool(settings, "advanced");
	for (const char *prop_name :
	     {"model_select", "useGPU", "numThreads", "rvm_quality", "threshold", "mask_expansion", "mask_every_x_frames"}) {
		obs_property_set_visible(obs_properties_get(ppts, prop_name), enabled);
	}
	return true;
}

obs_properties_t *background_filter_properties(void *data)
{
	UNUSED_PARAMETER(data);
	obs_properties_t *props = obs_properties_create();

	/* Background: 0 = transparent (show what's below the camera in the scene), otherwise blur */
	obs_properties_add_int_slider(props, "blur_background", obs_module_text("BlurBackgroundFactor0NoBlurUseColor"),
				      0, 20, 1);

	obs_property_t *advanced = obs_properties_add_bool(props, "advanced", obs_module_text("Advanced"));
	obs_property_set_modified_callback(advanced, enable_advanced_settings);

	obs_property_t *p_model_select = obs_properties_add_list(props, "model_select",
								 obs_module_text("SegmentationModel"),
								 OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_STRING);
	obs_property_list_add_string(p_model_select, obs_module_text("RobustVideoMatting"), MODEL_RVM);
	obs_property_list_add_string(p_model_select, obs_module_text("MediaPipe"), MODEL_MEDIAPIPE);

	obs_property_t *p_rvm_quality = obs_properties_add_list(props, "rvm_quality", obs_module_text("RVMQuality"),
								OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
	obs_property_list_add_int(p_rvm_quality, obs_module_text("RVMQualityFast"), 0);
	obs_property_list_add_int(p_rvm_quality, obs_module_text("RVMQualityBalanced"), 1);
	obs_property_list_add_int(p_rvm_quality, obs_module_text("RVMQualityHigh"), 2);

	obs_property_t *p_use_gpu = obs_properties_add_list(props, "useGPU", obs_module_text("InferenceDevice"),
							    OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_STRING);
	obs_property_list_add_string(p_use_gpu, obs_module_text("CoreMLGPU"), USEGPU_COREML_GPU);
	obs_property_list_add_string(p_use_gpu, obs_module_text("CPU"), USEGPU_CPU);

	obs_properties_add_int_slider(props, "numThreads", obs_module_text("NumThreads"), 0, 8, 1);

	obs_properties_add_float_slider(props, "threshold", obs_module_text("Threshold"), 0.0, 1.0, 0.025);
	obs_properties_add_int_slider(props, "mask_expansion", obs_module_text("MaskExpansion"), -30, 30, 1);
	obs_properties_add_int(props, "mask_every_x_frames", obs_module_text("CalculateMaskEveryXFrame"), 1, 300, 1);

	obs_properties_add_bool(props, "stop_when_source_is_inactive",
				obs_module_text("StopWhenInactive"));

	// replace the placeholder with the current version
	const std::string basic_info = std::regex_replace(PLUGIN_INFO_TEMPLATE, std::regex("%1"), PLUGIN_VERSION);
	obs_properties_add_text(props, "info", basic_info.c_str(), OBS_TEXT_INFO);

	return props;
}

void background_filter_defaults(obs_data_t *settings)
{
	obs_data_set_default_int(settings, "blur_background", 0);
	obs_data_set_default_bool(settings, "advanced", false);
	obs_data_set_default_string(settings, "model_select", MODEL_RVM);
	obs_data_set_default_int(settings, "rvm_quality", 1);
	obs_data_set_default_string(settings, "useGPU", USEGPU_COREML_GPU);
	obs_data_set_default_int(settings, "numThreads", 4);
	obs_data_set_default_double(settings, "threshold", 0.5);
	obs_data_set_default_int(settings, "mask_expansion", 0);
	obs_data_set_default_int(settings, "mask_every_x_frames", 1);
	obs_data_set_default_bool(settings, "stop_when_source_is_inactive", true);
}

/**
 * Scenes saved by the full plugin can hold a model or inference device this build no longer has
 * (SINet, PPHumanSeg, plain CoreML, CUDA...). Move them to the current defaults so the filter keeps
 * working instead of failing to load a model, and save the change back into the scene.
 */
static void migrate_removed_settings(obs_data_t *settings)
{
	const std::string model = obs_data_get_string(settings, "model_select");
	if (model != MODEL_RVM && model != MODEL_MEDIAPIPE) {
		obs_log(LOG_WARNING, "Model %s is not in this build; using Robust Video Matting", model.c_str());
		obs_data_set_string(settings, "model_select", MODEL_RVM);
	}
	const std::string device = obs_data_get_string(settings, "useGPU");
	if (device != USEGPU_CPU && device != USEGPU_COREML_GPU) {
		obs_log(LOG_WARNING, "Inference device %s is not in this build; using CoreML GPU", device.c_str());
		obs_data_set_string(settings, "useGPU", USEGPU_COREML_GPU);
	}
}

void background_filter_update(void *data, obs_data_t *settings)
{
	obs_log(LOG_INFO, "Background filter updated");

	// Cast to shared_ptr pointer and create a local shared_ptr
	auto *ptr = static_cast<std::shared_ptr<background_removal_filter> *>(data);
	if (!ptr) {
		return;
	}

	std::shared_ptr<background_removal_filter> tf = *ptr;
	if (!tf) {
		return;
	}

	tf->isDisabled = true;

	migrate_removed_settings(settings);

	tf->stopWhenSourceIsInactive = obs_data_get_bool(settings, "stop_when_source_is_inactive");
	tf->threshold = (float)obs_data_get_double(settings, "threshold");
	tf->maskExpansion = (int)obs_data_get_int(settings, "mask_expansion");
	tf->maskEveryXFrames = std::max(1, (int)obs_data_get_int(settings, "mask_every_x_frames"));
	tf->maskEveryXFramesCount = 0;
	tf->blurBackground = obs_data_get_int(settings, "blur_background");
	tf->rvmQuality = (int)obs_data_get_int(settings, "rvm_quality");

	const std::string newUseGpu = obs_data_get_string(settings, "useGPU");
	const std::string newModel = obs_data_get_string(settings, "model_select");
	const uint32_t newNumThreads = (uint32_t)obs_data_get_int(settings, "numThreads");

	if (tf->modelSelection.empty() || tf->modelSelection != newModel || tf->useGPU != newUseGpu ||
	    tf->numThreads != newNumThreads) {
		std::unique_lock<std::mutex> lock(tf->modelMutex);

		// Re-initialize the model when the model, inference device or thread count changes
		tf->modelSelection = newModel;
		tf->useGPU = newUseGpu;
		tf->numThreads = newNumThreads;

		if (tf->modelSelection == MODEL_MEDIAPIPE) {
			tf->model.reset(new ModelMediaPipe);
		} else {
			tf->model.reset(new ModelRVM);
		}
		// lastPersonAlpha belongs to the worker thread; a new model resets it there (configureForFrame)

		int ortSessionResult = createOrtSession(tf.get());
		if (ortSessionResult != OBS_BGREMOVAL_ORT_SESSION_SUCCESS) {
			obs_log(LOG_ERROR, "Failed to create ONNXRuntime session. Error code: %d", ortSessionResult);
			// disable filter
			tf->isDisabled = true;
			tf->model.reset();
			return;
		}
	}

	{
		std::unique_lock<std::mutex> lock(tf->modelMutex);
		if (auto *rvm = dynamic_cast<ModelRVM *>(tf->model.get())) {
			rvm->setQuality(tf->rvmQuality);
		}
	}

	obs_enter_graphics();

	char *effect_path = obs_module_file(EFFECT_PATH);
	gs_effect_destroy(tf->effect);
	tf->effect = gs_effect_create_from_file(effect_path, NULL);
	bfree(effect_path);

	char *kawaseBlurEffectPath = obs_module_file(KAWASE_BLUR_EFFECT_PATH);
	gs_effect_destroy(tf->kawaseBlurEffect);
	tf->kawaseBlurEffect = gs_effect_create_from_file(kawaseBlurEffectPath, NULL);
	bfree(kawaseBlurEffectPath);

	obs_leave_graphics();

	obs_log(LOG_INFO, "Background Removal Filter Options:");
	obs_log(LOG_INFO, "  Source: %s", obs_source_get_name(tf->source));
	obs_log(LOG_INFO, "  Model: %s", tf->modelSelection.c_str());
	obs_log(LOG_INFO, "  RVM Quality: %d", tf->rvmQuality);
	obs_log(LOG_INFO, "  Inference Device: %s", tf->useGPU.c_str());
	obs_log(LOG_INFO, "  Num Threads: %d", tf->numThreads);
	obs_log(LOG_INFO, "  Threshold: %f", tf->threshold);
	obs_log(LOG_INFO, "  Mask Expansion: %d", tf->maskExpansion);
	obs_log(LOG_INFO, "  Mask Every X Frames: %d", tf->maskEveryXFrames);
	obs_log(LOG_INFO, "  Blur Background: %d", (int)tf->blurBackground);
	obs_log(LOG_INFO, "  Model file path: %s", tf->modelFilepath.c_str());

	// fresh watchdog state for the new settings
	tf->memBaselineBytes = 0;
	tf->memWatchStartNs = 0;
	tf->memLastActionNs = 0;
	tf->memStrikes = 0;
	tf->memoryStopped = false;

	// enable
	tf->isDisabled = false;
}

void background_filter_activate(void *data)
{
	auto *ptr = static_cast<std::shared_ptr<background_removal_filter> *>(data);
	if (!ptr) {
		return;
	}

	std::shared_ptr<background_removal_filter> tf = *ptr;
	if (tf && tf->stopWhenSourceIsInactive) {
		obs_log(LOG_INFO, "Background filter activated");
		tf->isDisabled = false;
	}
}

void background_filter_deactivate(void *data)
{
	auto *ptr = static_cast<std::shared_ptr<background_removal_filter> *>(data);
	if (!ptr) {
		return;
	}

	std::shared_ptr<background_removal_filter> tf = *ptr;
	if (tf && tf->stopWhenSourceIsInactive) {
		obs_log(LOG_INFO, "Background filter deactivated");
		tf->isDisabled = true;
	}
}

/**                   FILTER CORE                     */

static void maskWorkerLoop(std::weak_ptr<background_removal_filter> weak);

void *background_filter_create(obs_data_t *settings, obs_source_t *source)
{
	obs_log(LOG_INFO, "Background filter created");
	try {
		// Create the instance as a shared_ptr
		auto instance = std::make_shared<background_removal_filter>();

		instance->source = source;
		instance->texrender = gs_texrender_create(GS_BGRA, GS_ZS_NONE);

		std::string instanceName{"background-removal-inference"};
		instance->env.reset(new Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR, instanceName.c_str()));

		instance->modelSelection = MODEL_MEDIAPIPE;

		// Create pointer to shared_ptr for the update call
		auto ptr = new std::shared_ptr<background_removal_filter>(instance);
		background_filter_update(ptr, settings);

		instance->worker = std::thread(maskWorkerLoop, std::weak_ptr<background_removal_filter>(instance));

		// Return the pointer to the shared_ptr
		// This keeps the reference count at least 1 until destroy is called
		return ptr;
	} catch (const std::exception &e) {
		obs_log(LOG_ERROR, "Failed to create background filter: %s", e.what());
		return nullptr;
	}
}

void background_filter_destroy(void *data)
{
	obs_log(LOG_INFO, "Background filter destroyed");

	// Cast back to shared_ptr pointer
	auto *ptr = static_cast<std::shared_ptr<background_removal_filter> *>(data);
	if (ptr) {
		if (*ptr) {
			// Mark as disabled to prevent further processing
			(*ptr)->isDisabled = true;

			// Stop the mask worker and wait for any mask in progress
			{
				std::lock_guard<std::mutex> lock((*ptr)->workMutex);
				(*ptr)->workerStop = true;
			}
			(*ptr)->workCv.notify_all();
			if ((*ptr)->worker.joinable()) {
				(*ptr)->worker.join();
			}

			// Perform cleanup
			obs_enter_graphics();
			gs_texrender_destroy((*ptr)->texrender);
			if ((*ptr)->stagesurface) {
				gs_stagesurface_destroy((*ptr)->stagesurface);
			}
			gs_effect_destroy((*ptr)->effect);
			gs_effect_destroy((*ptr)->kawaseBlurEffect);
			obs_leave_graphics();
		}
		// Delete the pointer to shared_ptr
		// This decrements the ref count. If no other threads hold a shared_ptr, the instance is deleted
		delete ptr;
	}
}

static void processImageForBackground(struct background_removal_filter *tf, const cv::Mat &imageBGRA,
				      cv::Mat &backgroundMask)
{
	// Models with dynamic input shapes (RVM) size themselves to the frame
	if (tf->model && tf->session && tf->model->configureForFrame((uint32_t)imageBGRA.cols, (uint32_t)imageBGRA.rows)) {
		tf->model->populateInputOutputShapes(tf->session, tf->inputDims, tf->outputDims);
		tf->model->allocateTensorBuffers(tf->inputDims, tf->outputDims, tf->outputTensorValues,
						 tf->inputTensorValues, tf->inputTensor, tf->outputTensor);
		tf->lastPersonAlpha.release();
	}

	cv::Mat outputImage;
	if (!runFilterModelInference(tf, imageBGRA, outputImage)) {
		return;
	}
	// outputImage: single channel uint8, 255 = person. refineMaskStable() does the thresholding.
	backgroundMask = 255 - outputImage;
}

/**
 * Fast guided filter (He et al.): smooths p while following the edges in the guide image I.
 * I and p are CV_32F single channel, same size.
 */
static cv::Mat guidedFilter(const cv::Mat &I, const cv::Mat &p, int r, double eps)
{
	const cv::Size k(2 * r + 1, 2 * r + 1);
	cv::Mat meanI, meanP, corrIP, corrII;
	cv::boxFilter(I, meanI, CV_32F, k);
	cv::boxFilter(p, meanP, CV_32F, k);
	cv::boxFilter(I.mul(p), corrIP, CV_32F, k);
	cv::boxFilter(I.mul(I), corrII, CV_32F, k);
	cv::Mat covIP = corrIP - meanI.mul(meanP);
	cv::Mat varI = corrII - meanI.mul(meanI);
	cv::Mat a, b;
	cv::divide(covIP, varI + eps, a);
	b = meanP - a.mul(meanI);
	cv::Mat meanA, meanB;
	cv::boxFilter(a, meanA, CV_32F, k);
	cv::boxFilter(b, meanB, CV_32F, k);
	return meanA.mul(I) + meanB;
}

/**
 * Stable mask refinement.
 *
 * Input: soft background mask at model resolution (255 = background).
 * Output: soft background mask at frame resolution.
 *
 * 1. Asymmetric temporal smoothing: the person is added quickly but removed slowly, so a one-frame
 *    dropout on the face does not punch a hole in it.
 * 2. Hysteresis: confident pixels (>= threshold + 0.2) seed the person; less confident pixels
 *    (>= threshold - 0.25) are kept only if they connect to a seed.
 * 3. Main subject: drop blobs smaller than 15% of the largest one (stray furniture, shelves).
 * 4. Fill holes inside the person that are smaller than 5% of the frame (face/shirt dropouts).
 * 5. Edge refinement: a guided filter on the camera image snaps the edge to hair and shoulders.
 */
static void refineMaskStable(struct background_removal_filter *tf, const cv::Mat &imageBGRA,
			     cv::Mat &backgroundMask)
{
	cv::Mat person;
	cv::Mat(255 - backgroundMask).convertTo(person, CV_32F, 1.0 / 255.0);

	// 1. temporal
	if (!tf->lastPersonAlpha.empty() && tf->lastPersonAlpha.size() == person.size()) {
		cv::Mat fast, slow;
		cv::addWeighted(person, 0.9, tf->lastPersonAlpha, 0.1, 0.0, fast);
		cv::addWeighted(person, 0.6, tf->lastPersonAlpha, 0.4, 0.0, slow);
		cv::Mat rising = person > tf->lastPersonAlpha;
		fast.copyTo(slow, rising);
		person = slow;
	}
	tf->lastPersonAlpha = person.clone();

	// 2. hysteresis
	const float hi = std::min(tf->threshold + 0.2f, 0.95f);
	const float lo = std::max(tf->threshold - 0.25f, 0.05f);
	cv::Mat strong = person >= hi;
	cv::Mat weak = person >= lo;

	cv::Mat labels, stats, centroids;
	const int n = cv::connectedComponentsWithStats(weak, labels, stats, centroids, 8, CV_32S);
	std::vector<uint8_t> keep((size_t)n, 0);
	for (int y = 0; y < labels.rows; y++) {
		const int32_t *l = labels.ptr<int32_t>(y);
		const uint8_t *s = strong.ptr<uint8_t>(y);
		for (int x = 0; x < labels.cols; x++) {
			if (s[x]) {
				keep[(size_t)l[x]] = 1;
			}
		}
	}
	keep[0] = 0;

	// 3. main subject
	int maxArea = 0;
	for (int i = 1; i < n; i++) {
		if (keep[(size_t)i]) {
			maxArea = std::max(maxArea, stats.at<int32_t>(i, cv::CC_STAT_AREA));
		}
	}
	for (int i = 1; i < n; i++) {
		if (keep[(size_t)i] && stats.at<int32_t>(i, cv::CC_STAT_AREA) < (int)(maxArea * 0.15)) {
			keep[(size_t)i] = 0;
		}
	}
	cv::Mat core(person.size(), CV_8UC1);
	for (int y = 0; y < labels.rows; y++) {
		const int32_t *l = labels.ptr<int32_t>(y);
		uint8_t *c = core.ptr<uint8_t>(y);
		for (int x = 0; x < labels.cols; x++) {
			c[x] = keep[(size_t)l[x]] ? 255 : 0;
		}
	}

	// 4. fill holes
	{
		cv::Mat inv = 255 - core;
		cv::Mat hl, hs, hc;
		const int hn = cv::connectedComponentsWithStats(inv, hl, hs, hc, 4, CV_32S);
		const int maxHole = (int)(core.total() * 0.05);
		std::vector<uint8_t> fill((size_t)hn, 0);
		for (int i = 1; i < hn; i++) {
			const int x = hs.at<int32_t>(i, cv::CC_STAT_LEFT);
			const int y = hs.at<int32_t>(i, cv::CC_STAT_TOP);
			const int w = hs.at<int32_t>(i, cv::CC_STAT_WIDTH);
			const int h = hs.at<int32_t>(i, cv::CC_STAT_HEIGHT);
			const int a = hs.at<int32_t>(i, cv::CC_STAT_AREA);
			if (x > 0 && y > 0 && x + w < core.cols && y + h < core.rows && a < maxHole) {
				fill[(size_t)i] = 1;
			}
		}
		for (int y = 0; y < hl.rows; y++) {
			const int32_t *l = hl.ptr<int32_t>(y);
			uint8_t *c = core.ptr<uint8_t>(y);
			for (int x = 0; x < hl.cols; x++) {
				if (fill[(size_t)l[x]]) {
					c[x] = 255;
				}
			}
		}
	}

	// soft alpha: zero outside a band around the core, solid inside it
	const int kk = std::max(3, (std::max(core.rows, core.cols) / 80) | 1);
	const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(kk, kk));
	cv::Mat band, inner;
	cv::dilate(core, band, kernel);
	cv::erode(core, inner, kernel);
	cv::Mat alpha = person.clone();
	alpha.setTo(0.0f, band == 0);
	alpha.setTo(1.0f, inner > 0);

	// 5. guided edge refinement at a capped working resolution
	const int frameW = imageBGRA.cols, frameH = imageBGRA.rows;
	const int workLong = std::min(std::max(frameW, frameH), 960);
	const double sc = (double)workLong / (double)std::max(frameW, frameH);
	const cv::Size workSize(std::max(1, (int)std::lround(frameW * sc)), std::max(1, (int)std::lround(frameH * sc)));

	cv::Mat alphaWork, grayWork, smallBGRA;
	cv::resize(alpha, alphaWork, workSize, 0, 0, cv::INTER_LINEAR);
	cv::resize(imageBGRA, smallBGRA, workSize, 0, 0, cv::INTER_AREA);
	cv::cvtColor(smallBGRA, grayWork, cv::COLOR_BGRA2GRAY);
	grayWork.convertTo(grayWork, CV_32F, 1.0 / 255.0);
	alphaWork = guidedFilter(grayWork, alphaWork, std::max(2, workLong / 240), 1e-3);

	cv::Mat alpha8;
	alphaWork.convertTo(alpha8, CV_8U, 255.0); // saturates to [0, 255]
	cv::resize(alpha8, alpha8, imageBGRA.size(), 0, 0, cv::INTER_LINEAR);

	if (tf->maskExpansion > 0) {
		cv::dilate(alpha8, alpha8, cv::Mat(), cv::Point(-1, -1), tf->maskExpansion);
	} else if (tf->maskExpansion < 0) {
		cv::erode(alpha8, alpha8, cv::Mat(), cv::Point(-1, -1), -tf->maskExpansion);
	}

	backgroundMask = 255 - alpha8;
}

/**
 * Physical memory footprint of the whole OBS process, in bytes (0 if unknown).
 */
static uint64_t processMemoryFootprintBytes()
{
	task_vm_info_data_t info;
	mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
	if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info, &count) == KERN_SUCCESS) {
		return (uint64_t)info.phys_footprint;
	}
	return 0;
}

/**
 * Memory watchdog.
 *
 * Records OBS's memory footprint 15 s after the filter starts. If the footprint later grows by more
 * than 3 GB, the filter assumes the inference backend is leaking and escalates, at least 10 s apart:
 *   1. rebuild the ONNX Runtime session (releases everything the backend allocated),
 *   2. if it grows again: switch this filter to CPU inference and rebuild,
 *   3. if it still grows: stop processing (the camera passes through unchanged).
 * Every step is logged. Changing any filter setting resets the watchdog.
 * This turns a runaway leak into, at worst, a few GB and a log entry instead of a frozen Mac.
 */
static void checkMemoryWatchdog(std::shared_ptr<background_removal_filter> tf)
{
	const uint64_t now = os_gettime_ns();
	const uint64_t second = 1000000000ULL;
	const uint64_t limitBytes = 3ULL * 1024 * 1024 * 1024;

	if (tf->memWatchStartNs == 0) {
		tf->memWatchStartNs = now;
		return;
	}
	if (now - tf->memLastCheckNs < 2 * second) {
		return;
	}
	tf->memLastCheckNs = now;

	const uint64_t footprint = processMemoryFootprintBytes();
	if (footprint == 0) {
		return;
	}
	if (tf->memBaselineBytes == 0) {
		if (now - tf->memWatchStartNs >= 15 * second) {
			tf->memBaselineBytes = footprint;
			obs_log(LOG_INFO, "Memory watchdog: baseline %.2f GB", footprint / 1073741824.0);
		}
		return;
	}
	if (footprint <= tf->memBaselineBytes + limitBytes) {
		return;
	}
	if (tf->memLastActionNs != 0 && now - tf->memLastActionNs < 10 * second) {
		return; // give the previous step time to take effect
	}
	tf->memLastActionNs = now;
	tf->memStrikes++;

	obs_log(LOG_WARNING, "Memory watchdog: OBS memory grew from %.2f GB to %.2f GB (step %d)",
		tf->memBaselineBytes / 1073741824.0, footprint / 1073741824.0, tf->memStrikes);

	if (tf->memStrikes >= 3) {
		obs_log(LOG_ERROR,
			"Memory watchdog: still growing after rebuilding and switching to CPU. Background removal "
			"is paused for this source; change any filter setting to resume.");
		tf->memoryStopped = true;
		return;
	}

	std::unique_lock<std::mutex> lock(tf->modelMutex);
	if (!tf->model) {
		return;
	}
	if (tf->memStrikes == 2 && tf->useGPU != USEGPU_CPU) {
		obs_log(LOG_WARNING, "Memory watchdog: switching inference device from %s to CPU",
			tf->useGPU.c_str());
		tf->useGPU = USEGPU_CPU;
		// one CPU thread is too slow for real-time RVM; use four
		tf->numThreads = std::max<uint32_t>(tf->numThreads, 4);
	} else {
		obs_log(LOG_WARNING, "Memory watchdog: rebuilding the inference session");
	}
	tf->model->resetTemporalState();
	tf->lastPersonAlpha.release();
	const int result = createOrtSession(tf.get());
	if (result != OBS_BGREMOVAL_ORT_SESSION_SUCCESS) {
		obs_log(LOG_ERROR, "Memory watchdog: failed to rebuild session (%d). Pausing background removal.",
			result);
		tf->memoryStopped = true;
	}
}

/**
 * Computes one mask from the latest captured frame. Runs on the mask worker thread.
 */
static void computeMask(std::shared_ptr<background_removal_filter> tf)
{
	if (tf->isDisabled || !tf->model) {
		return;
	}

	checkMemoryWatchdog(tf);
	if (tf->memoryStopped) {
		return;
	}

	// Skip frames when "Calculate every X frame" is above 1; the previous mask stays in place
	tf->maskEveryXFramesCount = (tf->maskEveryXFramesCount + 1) % tf->maskEveryXFrames;
	if (tf->maskEveryXFramesCount != 0) {
		return;
	}

	cv::Mat imageBGRA;
	{
		std::lock_guard<std::mutex> lock(tf->inputBGRALock);
		if (tf->inputBGRA.empty()) {
			// No data to process
			return;
		}
		imageBGRA = tf->inputBGRA.clone();
	}

	{
		// Until the first mask is ready, treat the whole frame as background so the room never shows
		std::lock_guard<std::mutex> lock(tf->outputLock);
		if (tf->backgroundMask.empty()) {
			tf->backgroundMask = cv::Mat(imageBGRA.size(), CV_8UC1, cv::Scalar(255));
		}
	}

	try {
		cv::Mat backgroundMask;
		{
			std::unique_lock<std::mutex> lock(tf->modelMutex);
			processImageForBackground(tf.get(), imageBGRA, backgroundMask);
		}
		if (backgroundMask.empty()) {
			obs_log(LOG_WARNING, "Background mask is empty. Using previous mask.");
			return;
		}

		refineMaskStable(tf.get(), imageBGRA, backgroundMask);

		std::lock_guard<std::mutex> lock(tf->outputLock);
		backgroundMask.copyTo(tf->backgroundMask);
	} catch (const Ort::Exception &e) {
		obs_log(LOG_ERROR, "ONNXRuntime Exception: %s", e.what());
	} catch (const std::exception &e) {
		obs_log(LOG_ERROR, "%s", e.what());
	}
}

/**
 * Mask worker thread. Waits for the graphics thread to signal a new frame, then computes the mask.
 * Frames that arrive while a mask is being computed are skipped, so slow inference lowers the mask
 * rate instead of stalling OBS's rendering.
 */
static void maskWorkerLoop(std::weak_ptr<background_removal_filter> weak)
{
	while (true) {
		std::shared_ptr<background_removal_filter> tf = weak.lock();
		if (!tf) {
			return;
		}
		{
			std::unique_lock<std::mutex> lock(tf->workMutex);
			tf->workCv.wait_for(lock, std::chrono::milliseconds(200),
					    [&] { return tf->workPending || tf->workerStop; });
			if (tf->workerStop) {
				return;
			}
			if (!tf->workPending) {
				continue;
			}
			tf->workPending = false;
		}
		computeMask(tf);
	}
}

void background_filter_video_tick(void *data, float seconds)
{
	UNUSED_PARAMETER(seconds);

	auto *ptr = static_cast<std::shared_ptr<background_removal_filter> *>(data);
	if (!ptr) {
		return;
	}
	std::shared_ptr<background_removal_filter> tf = *ptr;
	if (!tf || tf->isDisabled || tf->memoryStopped) {
		return;
	}
	if (!obs_source_enabled(tf->source)) {
		return;
	}

	// Hand the newest frame to the worker; never block the graphics thread on inference
	{
		std::lock_guard<std::mutex> lock(tf->workMutex);
		tf->workPending = true;
	}
	tf->workCv.notify_one();
}

/**
 * Blurs the background only: the mask-aware Kawase blur leaves pixels where the mask is 0 (the
 * person) untouched and blurs the rest using background pixels, so the person doesn't bleed into it.
 */
static gs_texture_t *blur_background(std::shared_ptr<background_removal_filter> tf, uint32_t width, uint32_t height,
				     gs_texture_t *alphaTexture)
{
	if (tf->blurBackground == 0 || !tf->kawaseBlurEffect) {
		return nullptr;
	}
	gs_texture_t *blurredTexture = gs_texture_create(width, height, GS_BGRA, 1, nullptr, 0);
	gs_copy_texture(blurredTexture, gs_texrender_get_texture(tf->texrender));
	gs_eparam_t *image = gs_effect_get_param_by_name(tf->kawaseBlurEffect, "image");
	gs_eparam_t *mask = gs_effect_get_param_by_name(tf->kawaseBlurEffect, "focalmask");
	gs_eparam_t *xOffset = gs_effect_get_param_by_name(tf->kawaseBlurEffect, "xOffset");
	gs_eparam_t *yOffset = gs_effect_get_param_by_name(tf->kawaseBlurEffect, "yOffset");

	for (int i = 0; i < (int)tf->blurBackground; i++) {
		gs_texrender_reset(tf->texrender);
		if (!gs_texrender_begin(tf->texrender, width, height)) {
			obs_log(LOG_INFO, "Could not open background blur texrender!");
			return blurredTexture;
		}

		gs_effect_set_texture(image, blurredTexture);
		gs_effect_set_texture(mask, alphaTexture);
		gs_effect_set_float(xOffset, ((float)i + 0.5f) / (float)width);
		gs_effect_set_float(yOffset, ((float)i + 0.5f) / (float)height);

		struct vec4 background;
		vec4_zero(&background);
		gs_clear(GS_CLEAR_COLOR, &background, 0.0f, 0);
		gs_ortho(0.0f, static_cast<float>(width), 0.0f, static_cast<float>(height), -100.0f, 100.0f);
		gs_blend_state_push();
		gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);

		while (gs_effect_loop(tf->kawaseBlurEffect, "Draw")) {
			gs_draw_sprite(blurredTexture, 0, width, height);
		}
		gs_blend_state_pop();
		gs_texrender_end(tf->texrender);
		gs_copy_texture(blurredTexture, gs_texrender_get_texture(tf->texrender));
	}
	return blurredTexture;
}

void background_filter_video_render(void *data, gs_effect_t *_effect)
{
	UNUSED_PARAMETER(_effect);

	// Cast to shared_ptr pointer and create a local shared_ptr
	auto *ptr = static_cast<std::shared_ptr<background_removal_filter> *>(data);
	if (!ptr) {
		return;
	}

	// Create a local shared_ptr
	std::shared_ptr<background_removal_filter> tf = *ptr;

	if (!tf || tf->isDisabled || tf->memoryStopped) {
		// pass the camera through unchanged
		if (tf && tf->source) {
			obs_source_skip_video_filter(tf->source);
		}
		return;
	}

	uint32_t width, height;
	if (!getRGBAFromStageSurface(tf.get(), width, height)) {
		if (tf->source) {
			obs_source_skip_video_filter(tf->source);
		}
		return;
	}

	if (!tf->effect) {
		// Effect failed to load, skip rendering
		if (tf->source) {
			obs_source_skip_video_filter(tf->source);
		}
		return;
	}

	gs_texture_t *alphaTexture = nullptr;
	{
		std::lock_guard<std::mutex> lock(tf->outputLock);

		if (tf->backgroundMask.empty()) {
			// No frame captured yet
			if (tf->source) {
				obs_source_skip_video_filter(tf->source);
			}
			return;
		}

		alphaTexture = gs_texture_create(tf->backgroundMask.cols, tf->backgroundMask.rows, GS_R8, 1,
						 (const uint8_t **)&tf->backgroundMask.data, 0);

		if (!alphaTexture) {
			obs_log(LOG_ERROR, "Failed to create alpha texture");
			if (tf->source) {
				obs_source_skip_video_filter(tf->source);
			}
			return;
		}
	}

	// Output the masked image
	gs_texture_t *blurredTexture = blur_background(tf, width, height, alphaTexture);

	if (!obs_source_process_filter_begin(tf->source, GS_RGBA, OBS_ALLOW_DIRECT_RENDERING)) {
		if (tf->source) {
			obs_source_skip_video_filter(tf->source);
		}
		gs_texture_destroy(alphaTexture);
		gs_texture_destroy(blurredTexture);
		return;
	}

	gs_eparam_t *alphamask = gs_effect_get_param_by_name(tf->effect, "alphamask");
	gs_eparam_t *blurredBackground = gs_effect_get_param_by_name(tf->effect, "blurredBackground");

	gs_effect_set_texture(alphamask, alphaTexture);

	if (tf->blurBackground > 0) {
		gs_effect_set_texture(blurredBackground, blurredTexture);
	}

	gs_blend_state_push();
	gs_reset_blend_state();

	const char *techName = (tf->blurBackground > 0) ? "DrawWithBlur" : "DrawWithoutBlur";

	obs_source_process_filter_tech_end(tf->source, tf->effect, 0, 0, techName);

	gs_blend_state_pop();

	gs_texture_destroy(alphaTexture);
	gs_texture_destroy(blurredTexture);
}
