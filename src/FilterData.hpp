// SPDX-FileCopyrightText: 2021-2026 Roy Shilkrot <roy.shil@gmail.com>
// SPDX-FileCopyrightText: 2023-2026 Kaito Udagawa <umireon@kaito.tokyo>
//
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef FILTERDATA_H
#define FILTERDATA_H

#include <obs-module.h>

#include <atomic>
#include <memory>
#include <mutex>
#include <string>

#include "models/Model.hpp"
#include "ort-utils/ORTModelData.hpp"

/**
 * State shared by the model code and the OBS filter: the ONNX Runtime session, the chosen model and
 * device, and the camera frame captured for the mask worker.
 */
struct filter_data : public ORTModelData {
	std::string useGPU;
	uint32_t numThreads = 0;
	std::string modelSelection;
	std::unique_ptr<Model> model;

	obs_source_t *source = nullptr;
	gs_texrender_t *texrender = nullptr;
	gs_stagesurf_t *stagesurface = nullptr;

	cv::Mat inputBGRA;

	std::atomic<bool> isDisabled{false};

	std::mutex inputBGRALock;
	std::mutex outputLock;

	std::string modelFilepath;
};

#endif /* FILTERDATA_H */
