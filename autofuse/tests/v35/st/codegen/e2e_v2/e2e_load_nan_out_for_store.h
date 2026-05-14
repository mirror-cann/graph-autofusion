#ifndef __TEST_E2E_LOAD_NAN_OUT_FOR_STORE_H__
#define __TEST_E2E_LOAD_NAN_OUT_FOR_STORE_H__
#include "ascendc_ir.h"

void LoadNanOutForStore_BeforeAutofuse(af::AscGraph &graph);
void LoadNanOutForStore_AfterInferOutput(af::AscGraph &graph);
void LoadNanOutForStore_AfterGetApiInfo(af::AscGraph &graph);
void LoadNanOutForStore_AfterScheduler(af::AscGraph &graph);
void LoadNanOutForStore_AfterQueBufAlloc(af::AscGraph &graph);
#endif
