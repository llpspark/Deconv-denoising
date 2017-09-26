clear;close all;
%% settings
folder = '/home/gt/DeconvNet-master_3392/data/DATA2/TURBID/Photo_3';
numfiles=5;
filenameprefixed='test_224_tubid_I6';

size_input = 224;
size_label = 224;
stride = 56;

%% initialization
data = zeros(size_input, size_input, 3, 1);
label = zeros(size_label, size_label, 3, 1);
count = 0;

%% generate data
    image = imread(fullfile(folder,'a6.jpg'));
%     image90=rot90(image);
    image1=imread(fullfile(folder,'1.jpg'));
    im_input = im2double(image(:, :, :));
    im_label = im2double(image1(:, :, :));
    
    [hei,wid,~] = size(im_label);

    for x = 1 : stride : hei-size_input+1
        for y = 1 :stride : wid-size_input+1
            
            subim_input = im_input(x : x+size_input-1, y : y+size_input-1,:);
            subim_label = im_label(x : x+size_input-1, y : y+size_input-1,:);
            
            count=count+1;
            data(:, :, :, count) = subim_input;
            label(:, :, :, count) = subim_label;
        end
    end

order = randperm(count);
data = data(:, :, :, order);
label = label(:, :, :, order); 

%% writing to HDF5
numperfile=ceil(count/numfiles);
chunksz = 2;
for fileno=1:numfiles
    filename=strcat(filenameprefixed,num2str(fileno),'.h5');
    totalct = 0;
    created_flag = false;
  for batchno = 1:floor(numperfile/chunksz)
      fprintf('file no. %d, batch no.%d\n',fileno,batchno);
    last_read=(fileno-1)*numperfile+(batchno-1)*chunksz;
    batchdata = data(:,:,:,last_read+1:last_read+chunksz); 
    batchlabs = label(:,:,:,last_read+1:last_read+chunksz);
    
    %store to hdf5
    startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,1,1,totalct+1]);
    curr_dat_sz = store2hdf5(filename, batchdata, batchlabs, ~created_flag, startloc, chunksz); 
    created_flag = true;
    totalct = curr_dat_sz(end);
  end
end
%display structure of the stored hdf5 file
for fileno=1:numfiles
    filename=strcat(filenameprefixed,num2str(fileno),'.h5');
    h5disp(filename);
end