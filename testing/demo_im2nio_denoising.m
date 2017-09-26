clear all; clc; close all;
addpath('/home/gt/llp-Deconv1/caffe/matlab');

% work on grey-level image
image = imread('1.jpg');
im_noi_ = imread('a16.jpg');

% run denoising
im_noi = im2double(im_noi_);
im_pure_denoi = netforward(im_noi);

im_denoi=im_noi-im_pure_denoi;
 % compute psnr and ssim
 PSNR_noi = csnr(im_noi*255,image,0,0);
 SSIM_noi = cal_ssim(im_noi*255,image,0,0);
 
 PSNR_denoi = csnr(im_denoi*255,image,0,0);
 SSIM_denoi = cal_ssim(im_denoi*255,image,0,0);

% show results
figure
hold on
subplot(1,3,1); imshow(image);
subplot(1,3,2); imshow(uint8(im_noi*255));
subplot(1,3,3); imshow(uint8(im_denoi*255));
imwrite(im_denoi,'im2noi_vgg_a14.jpg')
hold off
