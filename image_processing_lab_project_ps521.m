clc, clearvars, close all


% find all file png files
D=dir('images/*.png');

score = [];

%load and process each file in turn.
for ind=1:length(D)

    %name of png file
    filename = fullfile(D(ind).folder,D(ind).name);

    %name of answer file .mat

    [folder, baseFileName, ~] = fileparts(filename);
    mat_filename = fullfile(folder, sprintf('%s.mat',baseFileName));

    %test result
    
    % load wrong file to test every for error detection
    if mod(ind,5)==0
        filename = fullfile(D(3).folder,D(1).name);
    end
    
    %call the actual findColours function - this is the function that the 
    % student needs to write
    try
        res = findColours(filename);
    catch ME
    % display the error message
    disp(['Error occurred: ', ME.message]);
    end

    % check the answers.
    mm = check_answer(res,mat_filename);

    score=[score,mm];
    %if mm ~= 100.00
    %    mat_filename
    %end

end
%print out the score.
str=repmat('%.2f ', 1, length(score));
fprintf('Score is: ');
fprintf(str,score);
fprintf('\nMean score %f\n',mean(score));


%s = findColours('images/org_1.png');

function colorMatrix = findColours(filename)
% Function to find the colors of the squares in the image
% Inputs:
%    filename - string variable that contains the path and filename
% Outputs:
%    colorMatrix - a 4x4 matrix containing the colors of each square in the
%    image

% Load the raw image file
image = loadImage(filename);
%figure(1)
%imshow(image)
% Find the coordinates of the 4 edge circles in the image
circleCoordinates = findCircles(image);
% Project the rotated image back to its original space
% based on the standard centroid coordinates
%figure(2)
correctedImage = correctImage(circleCoordinates,image);
%imshow(correctedImage)
% Calculate the centroid coordinates of each square in a standard unrotated
% image
referenceSquareCenters = findSquareCenters();

% Perform mean filtering on the image to remove salt and pepper and any
% other type of noise
image_dn = meanFilterImage(correctedImage, 6);

% Display the corrected, denoised image with the square centroids
% highlighted
% figure(21)
% imshow(image_dn)
% hold on;
% scatter(referenceSquareCenters(:,1), referenceSquareCenters(:,2), 'w', 'filled');
% hold off;
colorMatrix = detectSquareColours(image_dn, referenceSquareCenters);
colorMatrix
% imshow(image_dn, []);
end


function colorImageDouble = loadImage(filename)
% Function to load the image from a given filename
% Inputs:
%    filename- string containing the path and filename of the image
% Outputs:
%    colorImageDouble: The RGB image converted into double format
colorImage = imread(filename);
% convert the image to double
colorImageDouble = im2double(colorImage);
end


function centroids = findCircles(image)
% Function to find the 4 circles denoting the 4 corners of the image
% Inputs:
%    image: a double representation of the RGB image
% Outputs:
%    centroids: An array containing the x & y coordinates of the 4 circle
%    centroids in the given image

% Convert the image to grayscale
grayImage = rgb2gray(image);
% Apply a median filter to remove any remaining noise
filteredImage = medfilt2(grayImage);
% Automatically calculate the threshold from the image
% and use it to convert the image into black and white
threshold = graythresh(filteredImage);
binaryImage = imbinarize(filteredImage, threshold);

% Invert the binary image so that black pixels are now white and vice versa
invertedImage = imcomplement(binaryImage);
% Fill in any holes in the binary image
filledImage = imfill(invertedImage, 'holes');

% Use built-in MATLAB function to detect circles
% Second parameter of the function determines the minimum and maximum radii
% of the circles
% The function will detect circles brighter than the background and
% sensitivity is set high to detect maximum circles.
[centers, radii] = imfindcircles(filledImage, [10 200], 'ObjectPolarity', 'bright', 'Sensitivity', 0.85);

% Create binary image with detected circles
blackCirclesInImage = zeros(size(grayImage));
for i = 1:length(radii)
    % This creates a matrix representing the coordinates of each pixel in
    % the image and computes if the pixel belongs inside the image
    % If it belongs inside the image, the pixel is marked 1
    [x, y] = meshgrid(1:size(grayImage, 2), 1:size(grayImage, 1));
    blackCirclesInImage = blackCirclesInImage | ((x - centers(i, 1)).^2 + (y - centers(i, 2)).^2) <= radii(i)^2;
end

%figure(310)
%performs morphological area filtering and removes all connected components
%that has an area smaller than 4 pixels
circleImage = bwareafilt(blackCirclesInImage,4);
circle_s = regionprops(circleImage,'centroid');
%get the centroids of the 4 circles
centroids = cat(1,circle_s.Centroid);
%imshowpair(image, circle_img, 'blend')
%hold on;
%scatter(centroids(:,1), centroids(:,2), 'r', 'filled');
%hold off;

% Sort the centroids based on their distance from the top left corner
% This is to make sure all centroids are returned in order
% If centroid coordinates are not returned in order, then the geotrans
% and imwarp functions will fail to correct the image.
[~, idx] = sort(sum(centroids .^ 2, 2));
centroids = centroids(idx, :);
end



function correctedImage = correctImage(circleCoordinates, image)
%Function used to project the rotated, translated image back into its
%original position
%load a reference image to use as baseline to get circle coordinates
%In this case, we choose the org_1.png image to act as a standard image
%since this contains only small amount of noise and is not rotated.
referenceImage = loadImage('images/org_1.png');
referenceImageDouble = im2double(referenceImage);

%Circle centroid coordinates of the reference image.
referenceCircleCoordinates = findCircles(referenceImageDouble);

%Circle centroid coordinates of the current image being processed
currentCentres = im2double(circleCoordinates);

% find the transform matrix using these two sparse images
mytform = fitgeotrans(currentCentres,referenceCircleCoordinates,'projective');
    
%figure(8)
%imshow(image)
%figure(9)
%imshow(referenceImageDouble)
% correct the distorted image using the transform matrix found above
correctedImage = imwarp(image,mytform,'OutputView',imref2d(size(referenceImage)));
%figure(10)
%imshow(correctedImage)
%figure(11)

%Display the input image and corrected image one on top of the other.
imshowpair(image, correctedImage, 'blend')
end


function meanFilteredImage = meanFilterImage(image, filterSize)
% Function to perform mean filtering on the image. Results in a denoised
% image
% Inputs:
%   image - The double image containing noise
%   filterSize - This indicates the number of pixels used to calculate the
%   mean
% Outputs:
% meanFilteredImage - Denoised, mean filtered image
meanFilter = fspecial('average', filterSize);
meanFilteredImage = imfilter(image, meanFilter);
end


function squareCenters = findSquareCenters()
% This function identifies the squares arranged in 4x4 fashion in a
% reference image and returns the coordinates of the centres of all squares
% in order
% Inputs:
%   None
% Outputs:
%   squareCenters: An array containing the x and y coordinates of all the
%   squares in the reference image
referenceImage = imread('images/org_1.png');
meanFilteredImage = meanFilterImage(referenceImage, 4);
% Set a threshold mask that considers pixels that have an intensity above
% the threshold value of 35
thresholdMask = rgb2gray(meanFilteredImage)>35;
% figure(1)
% imshow(thresholdMask)

% Removes small, isolated bright regions in the thresholdMask.
thresholdMask = bwareaopen(thresholdMask,100); 
% Remove small, isolated dark regions in the thresholdMask
thresholdMask = ~bwareaopen(~thresholdMask,100); 

% Remove edge effects
% Removes the outermost connected component in the binary image
thresholdMask = imclearborder(thresholdMask); 
% Performs morphological erosion on the thresholdMask using a 10x10 square structuring element
thresholdMask = imerode(thresholdMask,ones(10)); 
%figure(5)
%imshow(thresholdMask)

% Label each square in the image
L = bwlabel(thresholdMask);
% Get the centroids of each labeled square
s = regionprops(L, 'centroid');
centroids = cat(1, s.Centroid);

% Sort the centroids of each square by row and then by column
[~, sorted_indices] = sort(centroids(:, 2)*size(thresholdMask, 1) + centroids(:, 1));
centroids_sorted = centroids(sorted_indices, :);

% Display the sorted centroids
squareCenters = centroids_sorted;
end


function color_array = detectSquareColours(rgbImage, squareCenters)
% Detects the color of each square arranged in 4x4 fashion
% in an RGB image by computing the mean L*a*b* values of a square region
% around each center coordinate.
% Inputs:
%   - rgbImage: a double RGB image matrix
%   - squareCenters: a 16x2 matrix containing the x and y coordinates of
%     the center of each square, in the order left to right, top to bottom
% Outputs:
%   - color_array: a 4x4 cell array containing the name of the color of each square

% Convert the RGB image to LAB color space for better color detection
labSpaceImage = rgb2lab(rgbImage);
%lab_image = imfilter(lab_image, meanFilter);
%lab_image = imadjust(lab_image, [0.1, 0.9],[]);

%figure(80)
%imshow(labSpaceImage)

% Define the max and min thresholds for the L,A,B channels for each color
% These are identified through trial and error
red_min_thresh = [45	62	35];
red_max_thresh = [58	83	67];
green_min_thresh = [74	-89	65];
green_max_thresh = [90	-70	86];
yellow_min_thresh = [80	-34 75];
yellow_max_thresh = [95	-14	96.54];
blue_min_thresh = [28	60	-105];
blue_max_thresh = [46	85	-70];
white_min_thresh = [85	-5	-5];
white_max_thresh = [102	5	5];

% Initialize the output color array
color_array = cell(4, 4);

% Define the size of the square region to use for color detection
square_size = 50;

% Loop over each square center and detect its color
for i = 1:size(squareCenters, 1)
    % Extract the square region around the center
    x_center = round(squareCenters(i, 1));
    y_center = round(squareCenters(i, 2));
    square_region = labSpaceImage(max(1,y_center-square_size/2):min(size(rgbImage,1),y_center+square_size/2), ...
                               max(1,x_center-square_size/2):min(size(rgbImage,2),x_center+square_size/2), :);

    % Compute the mean L*a*b* values of the square region
    lab_values = mean(mean(square_region, 1), 2);
    lab_values = reshape(lab_values, 1, 3);
    %squareCenters(i,1)
    %squareCenters(i,2)
    %lab_values
    
    % Determine the color of the square based on the color thresholds
    if all(all(lab_values >= red_min_thresh) & all(lab_values <= red_max_thresh))
        color_array{i} = 'red';
    elseif all(all(lab_values >= green_min_thresh) & all(lab_values <= green_max_thresh))
        color_array{i} = 'green';
    elseif all(all(lab_values >= yellow_min_thresh) & all(lab_values <= yellow_max_thresh))
        color_array{i} = 'yellow';
    elseif all(all(lab_values >= blue_min_thresh) & all(lab_values <= blue_max_thresh))
        color_array{i} = 'blue';
    elseif all(all(lab_values >= white_min_thresh) & all(lab_values <= white_max_thresh))
        color_array{i} = 'white';
    else
        color_array{i} = 'unknown';
    end
end

% Reshape the color array into a 4x4 matrix
color_array = transpose(reshape(color_array, 4, 4));
end










