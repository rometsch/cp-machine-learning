% Commands mostly taken from
% http://www.machinelearning.ru/wiki/images/e/ee/ReadMNIST.zip and
% modified for own purpose.
%% Load MNIST test set images.
filepath = './MNIST/t10k-images-idx3-ubyte';
fid = fopen(filepath,'r','b');      % big-endian
magicNum = fread(fid,1,'int32');
if(magicNum~=2051) 
    display('Error: cant find magic number');
    return;
end
imgNum = fread(fid,1,'int32');  % number of images in the set
rowSz = fread(fid,1,'int32');   % number of rows
colSz = fread(fid,1,'int32');   % number of colums

for k=1:imgNum
    % Transpose image to get the right orientation for plots.
    img = transpose(uint8(fread(fid,[rowSz colSz],'uchar')));
    % We want to store images as vectors.
    I{k} = reshape(img,[1,28*28]);
    % Plot with imagesc(reshape(img,28,28))
end

fclose(fid)
%% Load MNIST test set labels.
filepath = './MNIST/t10k-labels-idx1-ubyte';
fid = fopen(filepath,'r','b');      % big-endian
magicNum = fread(fid,1,'int32');
if(magicNum~=2049) 
    display('Error: cant find magic number');
    return;
end
itmNum = fread(fid,1,'int32');  % number of images in the set


labels = uint8(fread(fid,itmNum,'uint8'));


fclose(fid)
