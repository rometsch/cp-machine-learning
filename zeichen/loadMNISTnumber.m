function [ images ] = loadMNISTnumber( number )
%LOADMNISTNUMBER Load images of a given number from training set.
    
    % Test input.
    if (number<0) or (number>9)
        display('Error: Unable to load training set. Choose a number between 0 and 9.');
        return
    end
    
    % Commands mostly taken from
    % http://www.machinelearning.ru/wiki/images/e/ee/ReadMNIST.zip and
    % modified for own purpose.
    %% Load MNIST test set images.
    filepath = './MNIST/train-images-idx3-ubyte';
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

    fclose(fid);
    %% Load MNIST test set labels.
    filepath = './MNIST/train-labels-idx1-ubyte';
    fid = fopen(filepath,'r','b');      % big-endian
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
            images = [images ; I{n}];
        end
    end

    fclose(fid);
end

