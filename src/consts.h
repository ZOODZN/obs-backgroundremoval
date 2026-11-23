/*
 * SPDX-FileCopyrightText: 2021-2026 Roy Shilkrot <roy.shil@gmail.com>
 * SPDX-FileCopyrightText: 2023-2026 Kaito Udagawa <umireon@kaito.tokyo>
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef CONSTS_H
#define CONSTS_H

const char *const MODEL_RVM = "models/rvm_mobilenetv3_fp32.with_runtime_opt.ort";
const char *const MODEL_MEDIAPIPE = "models/mediapipe.with_runtime_opt.ort";

const char *const USEGPU_CPU = "cpu";
const char *const USEGPU_COREML_GPU = "coreml_gpu";

const char *const EFFECT_PATH = "effects/mask_alpha_filter.effect";
const char *const KAWASE_BLUR_EFFECT_PATH = "effects/kawase_blur.effect";

const char *const PLUGIN_INFO_TEMPLATE =
	"<a href=\"https://github.com/royshil/obs-backgroundremoval/\">Background Removal</a> (%1) by "
	"<a href=\"https://github.com/royshil\">Roy Shilkrot</a> ❤️ "
	"<a href=\"https://github.com/sponsors/royshil\">Support & Follow</a><br>"
	"ZOODESIGN build: RVM quality, stable mask, memory safeguard";

#endif /* CONSTS_H */
