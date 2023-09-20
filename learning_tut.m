clc, clearvars, close all

first_noise = imread('Assignment/images/noise_1.png');
figure(1)
imshow(first_noise)

%create a SRGB to Lab space filter
lab_filter = makecform('srgb2lab');
% apply this filter to convert to LAB space
lab_img = applycform(first_noise, lab_filter);
% convert image to a type double
lab_img_double = lab2double(lab_img);
figure(2)
imshow(lab_img)

figure(3)
imshow(lab_img_double)

h = fspecial('average', 5);
img_filtered = imfilter(lab_img, h);
figure(4)
imshow(img_filtered)

grey_img = im2gray(img_filtered);
figure(5)
imshow(grey_img)

invert_image = imcomplement(grey_img);
figure(6)
imshow(invert_image);

figure(7)
imhist(invert_image)

invert_image_adjusted = imadjust(invert_image);
figure(8)
imshow(invert_image_adjusted)



filter=fspecial('average',6);
denoised_img=imfilter(lab_img_double,filter);
L = denoised_img(:,:,1);
figure(9), 
imshow(L, [])
    
% erode and then dilate the image
im1=imerode(L,ones(7));
im2=imdilate(im1,ones(7));
im3=imerode(im2,ones(7));
figure(10)
imshow(im2)
% threshold 
im4 = im3>33;
figure(11)
imshow(im3)
%figure(2), imshow(im3, []);
figure(12)
bw = imcomplement(im4);
imshow(bw)

figure(13)
BW_matrix = bwareafilt(bw,1);
imshow(BW_matrix)
BW_circle = bwareafilt(bw,5);
figure(14)
imshow(BW_circle)


figure(15)
output_circ = imsubtract(BW_circle, BW_matrix);
imshow(output_circ)

s = regionprops(output_circ,'centroid');
centroids = cat(1,s.Centroid);
 
figure(16)
imshow(BW_circle)
hold on
plot(centroids(:,1),centroids(:,2),'b*');
hold off
circleCoordinates = centroids;
circleCoordinates