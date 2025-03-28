import requests
import os

API_URL = "http://localhost:8000/generate-video/"

def upload_files(prompt_image_path, canny_edges_paths):
    with open(prompt_image_path, "rb") as prompt_file:
        files = {"prompt_image": prompt_file}

        # Open multiple canny edge files
        canny_edge_files = [("canny_edges", open(path, "rb")) for path in canny_edges_paths]

        try:
            response = requests.post(API_URL, files={**files, **dict(canny_edge_files)})
        finally:
            # Ensure all files are closed
            for _, f in canny_edge_files:
                f.close()

        if response.status_code == 200:
            frames = response.json()["frames"]
            save_frames(frames)
        else:
            print(f"Error: {response.text}")

def save_frames(frames):
    os.makedirs("output_frames", exist_ok=True)
    for i, frame in enumerate(frames):
        frame_path = f"output_frames/frame_{i}.png"
        with open(frame_path, "wb") as f:
            f.write(bytearray(frame))
        print(f"Saved {frame_path}")

if __name__ == "__main__":
    prompt_image = "/shares/CC_v_Val_FV_Gen3_all/VIDT_DL/data/cnn_training/projects/smart_data_selection/model_and_test_data/cityscapes/CameraCalibration/RealCar3dDataset/2024_04_11_15_44_11/frame_00000.jpg"
    canny_edges = [
        "/shares/CC_v_Val_FV_Gen3_all/VIDT_DL/data/cnn_training/projects/smart_data_selection/model_and_test_data/cityscapes/CameraCalibration/RealCar3dDataset/2024_04_11_15_44_11/frame_00077_canny_edge.jpg",
        "/shares/CC_v_Val_FV_Gen3_all/VIDT_DL/data/cnn_training/projects/smart_data_selection/model_and_test_data/cityscapes/CameraCalibration/RealCar3dDataset/2024_04_11_15_44_11/frame_00115_canny_edge.jpg"
    ]

    upload_files(prompt_image, canny_edges)
