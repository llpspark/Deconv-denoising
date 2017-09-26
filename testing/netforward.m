function [ img_out ] = netforward(img)

[wid,hei,chn] = size(img);
caffe.reset_all();
caffe.set_mode_cpu();

model = '/home/gt/llp-Deconv1/training/im2noi_vgg_decon_newresi6/result/stage_2__iter_100000.caffemodel'; 

net = caffe.Net('/home/gt/llp-Deconv1/training/im2noi_vgg_decon_newresi6/llp_deploy.prototxt',model,'test');


img_out = zeros(wid, hei, chn,8);
for i = 1 : 4
    output = net.forward({rot90(img,i-1)});
    img_out(:,:,:,i) = rot90(output{1}, 5-i);
end
for i = 5 : 8
    output = net.forward({rot90(fliplr(img),i-5)});
    img_out(:,:,:,i) = fliplr(rot90(output{1}, 9-i));
end

img_out = mean(img_out,4);


end
