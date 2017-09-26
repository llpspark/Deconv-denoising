clear all; clc; close all;
addpath('/home/gt/llp-Deconv1/caffe/matlab');

% work on grey-level image
image = imread('3.jpg');
im_noi_ = imread('3a16.jpg');
im_noi_br=rgb2ycbcr(im_noi_);

% run denoising
im_noi = im2double(im_noi_);
im_denoi1 = netforward(im_noi);

im_denoi2=uint8(im_denoi1*255);
im_denoi3=rgb2ycbcr(im_denoi2);
im_denoi3(:,:,2)=im_noi_br(:,:,2);
im_denoi3(:,:,3)=im_noi_br(:,:,3);
im_denoi=ycbcr2rgb(im_denoi3);

 % compute psnr and ssim
 PSNR_noi = csnr(im_noi*255,image,0,0);
 SSIM_noi = cal_ssim(im_noi*255,image,0,0);
 
 PSNR_denoi = csnr(im_denoi,image,0,0);
 SSIM_denoi = cal_ssim(im_denoi,image,0,0);

% show results
figure
hold on
subplot(1,3,1); imshow(image);
subplot(1,3,2); imshow(uint8(im_noi*255));
subplot(1,3,3); imshow(uint8(im_denoi));
imwrite(im_denoi,'vgg_newresi6_3a16.jpg')
hold off