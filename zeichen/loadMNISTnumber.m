function [ images ] = loadMNISTnumber( type, number )
%LOADMNISTNUMBER Load images of a given number from either test or training set.
% MNIST data must be in a subfolder called 'MNIST'.

    %% Test input.
    if (number<0) || (number>9)
        display('Error: Unable to load training set. Choose a number between 0 and 9.');
        return
    end
    
    if not(strcmp(type,'train')) && not(strcmp(type,'test'))
        display('Error: Please pass either test or train as type argument.');
        return
    end

    %% Choose set type.
    if strcmp(type,'train')
        imagepath = './MNIST/train-images-idx3-ubyte';
        labelpath = './MNIST/train-labels-idx1-ubyte';
    end
    if strcmp(type,'test')
        imagepath = './MNIST/t10k-images-idx3-ubyte';
        labelpath = './MNIST/t10k-labels-idx1-ubyte';
    end


    %% Load MNIST test set images.
    
    % Commands for binary access of file taken from
    % http://www.machinelearning.ru/wiki/images/e/ee/ReadMNIST.zip and
    % modified for own purpose.
    
    fid = fopen(imagepath,'r','b');      % big-endian
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

    fclose(fid);
    %% Load MNIST test set labels.
    
    fid = fopen(labelpath,'r','b');      % big-endian
    magicNum = fread(fid,1,'int32');
    if(magicNum~=2049)
        display('Error: cant find magic number');
        return;
    end
    itmNum = fread(fid,1,'int32');  % number of images in the set


    labels = uint8(fread(fid,itmNum,'uint8'));

    
    %% Construct matrix with images of given number as rows.
    images = [];
    for n=1:1:imgNum
        if (labels(n)==number)
            images = [images , transpose(I{n})];
        end
    end
    % Cast image vectors to double for later use.
    images = double(images);
    
    fclose(fid);
end

