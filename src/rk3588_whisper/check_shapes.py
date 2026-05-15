from rknnlite.api import RKNNLite

def inspect_model(model_path):
    print(f"--- Inspecting {model_path} ---")
    rknn = RKNNLite()
    ret = rknn.load_rknn(model_path)
    if ret != 0:
        print("Failed to load model")
        return
    ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
    if ret != 0:
        print("Failed to init runtime")
        return
    
    # We can get inputs and outputs properties by accessing internal attributes or passing dummy data
    # Actually, RKNNLite doesn't have an easy public API for shapes without running, 
    # but we can try to print internal info if available.
    # Alternatively, we can look at the SDK's get_model_info or similar, but rknnlite usually lacks it.
    pass

# Actually, a better way is to parse the RKNN file header or run it with dummy arrays to see errors, 
# or just print out the expected inputs if the user exported it.
