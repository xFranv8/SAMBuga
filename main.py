import cv2
import numpy as np
from SAM import SAM
import time


def frames_to_video(frames: list, output_path: str, fps: int = 30):
    height, width, layers = frames[0].shape
    size = (width, height)

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    for frame in frames:
        out.write(frame)

    out.release()


def main():
    # Read video
    cap = cv2.VideoCapture("inputs/2_conduccion_saliendo_sotano_vuelta_por_campus.mp4")
    segmentator: SAM = SAM(model_type="vit_h", model_path="models/sam_vit_h_4b8939.pth", device="cuda")

    segmentated_frames = []
    while cap.isOpened():
        t0: float = time.time()
        ret, frame = cap.read()
        if ret:
            frame = frame[:, :frame.shape[1] // 2, :]
            frame = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))
            result: np.ndarray = segmentator.predict(frame)
            t1: float = time.time()

            cv2.putText(result, f"FPS: {round(1 / (t1 - t0), 2)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            segmentated_frames.append(result.astype(np.uint8))
            # cv2.imshow('frame', frame)
            # cv2.imshow('result', result.astype(np.uint8))
            #
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        else:
            break
    frames_to_video(segmentated_frames, "results/result.mp4", fps=30)
    # i: int = 0
    # for frame in segmentated_frames:
    #     cv2.imwrite(f"results/img{i}.jpg", frame)
    #     i += 1


if __name__ == '__main__':
    main()