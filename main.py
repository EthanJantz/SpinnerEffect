import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def make_ring_masks(width, height, n_rings):
    center = (width // 2, height // 2)

    max_radius = min(center)

    edges = np.linspace(0, max_radius, n_rings + 1).astype(int)

    masks = []
    for i in range(n_rings):
        r_inner, r_outer = edges[i], edges[i + 1]
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(mask, center, r_outer, 255, thickness=-1)
        cv2.circle(mask, center, r_inner, 0, thickness=-1)
        masks.append(mask)

    return masks


def main():
    parser = argparse.ArgumentParser(
        prog="SpinnerMask",
        description="A python implementation of a mask effect",
    )

    parser.add_argument("filename")
    parser.add_argument("-s", "--show", action="store_true")
    args = parser.parse_args()

    img = cv2.imread(args.filename)
    assert img is not None, "Must provide filename"

    height, width = img.shape[:2]
    center = (width // 2, height // 2)

    masks = make_ring_masks(width, height, n_rings=5)

    n_frames = 360
    frames = []
    pil_frames = []
    for frame in range(n_frames):
        canvas = img.copy()
        for idx, mask in enumerate(masks):
            direction = idx % 2 or -1
            angle, scale = frame * direction, 1.0
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
            rotated_image = cv2.warpAffine(img, rotation_matrix, (width, height))

            canvas[mask > 0] = rotated_image[mask > 0]
        pil_frames.append(Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)))
        frames.append(canvas)

    out_filename = Path(args.filename).stem
    pil_frames[0].save(
        f"{out_filename}_spinner.gif",
        save_all=True,
        append_images=pil_frames[1:],
        duration=33,
        loop=0,
    )

    if args.show:
        print("Press 'q' to exit")
        f = 0
        while True:
            cv2.imshow("anim", frames[f])
            f = (f + 1) % len(frames)
            if cv2.waitKey(33) == ord("q"):
                break
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
