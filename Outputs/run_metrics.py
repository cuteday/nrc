import os
from math import sqrt

import numpy as np

import cv2
from skimage import metrics, measure

class Params:
    evaluate_hdr = False
    evaluate_psnr = not evaluate_hdr
    base_folder = './Image'
    base_filename = 'zeroday'
    n_images = 50
    n_skip_frames = 30

if __name__ == "__main__":
    file_suffix = 'exr' if Params.evaluate_hdr else 'png'
    mrses = []
    psnrs = []

    print(Params.n_images)
    for i in range(Params.n_images):
        frame_count = i * Params.n_skip_frames

        path_ref = os.path.join(Params.base_folder, "GT", f"{Params.base_filename}_ref_{i:04d}.TonemappingPass.dst.{frame_count}.{file_suffix}")
        path_cp1 = os.path.join(Params.base_folder, "NRC", f"{Params.base_filename}_nrc_{i:04d}.NRCToneMapped.dst.{frame_count}.{file_suffix}")
        path_cp2 = os.path.join(Params.base_folder, "PT", f"{Params.base_filename}_pt_{i:04d}.TonemappingPass.dst.{frame_count}.{file_suffix}")

        if Params.evaluate_hdr:
            path_ref = os.path.join(Params.base_folder, "GT", f"{Params.base_filename}_ref_{i:04d}.AccumulatePass.output.{frame_count}.{file_suffix}")
            path_cp1 = os.path.join(Params.base_folder, "NRC", f"{Params.base_filename}_nrc_{i:04d}.NRCPathTracer.result.{frame_count}.{file_suffix}")
            path_cp2 = os.path.join(Params.base_folder, "PT", f"{Params.base_filename}_pt_{i:04d}.MegakernelPathTracer.color.{frame_count}.{file_suffix}")

        # print(path_ref)

        img_ref = cv2.imread(path_ref, cv2.IMREAD_UNCHANGED)
        img_cp1 = cv2.imread(path_cp1, cv2.IMREAD_UNCHANGED)
        img_cp2 = cv2.imread(path_cp2, cv2.IMREAD_UNCHANGED)

        # print(img_ref)

        mrse1 = sqrt(metrics.mean_squared_error(img_ref, img_cp1))
        mrse2 = sqrt(metrics.mean_squared_error(img_ref, img_cp2))
        mrses.append([mrse1, mrse2])

        if Params.evaluate_psnr:
            psnr1 = metrics.peak_signal_noise_ratio(img_ref, img_cp1)
            psnr2 = metrics.peak_signal_noise_ratio(img_ref, img_cp2)
        else: psnr1, psnr2 = 0., 0.
        psnrs.append([psnr1, psnr2])

        print(f"Image {i:02d} MRSE1 {mrse1} MRSE2 {mrse2} PSNR1 {psnr1} PSNR2 {psnr2}")
        # break
    
    avg_msre1 = sum([v[0] for v in mrses]) / Params.n_images
    avg_msre2 = sum([v[1] for v in mrses]) / Params.n_images
    avg_psnr1 = sum([v[0] for v in psnrs]) / Params.n_images
    avg_psnr2 = sum([v[1] for v in psnrs]) / Params.n_images

    print("Averaged results:")
    print(f"MRSE1 {avg_msre1} MRSE2 {avg_msre2}")
    print(f"PSNR1 {avg_psnr1} PSNR2 {avg_psnr2}")