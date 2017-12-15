%    a = VideoReader('IMG_0069.MOV');
%    for img = 1:a.NumberOfFrames;
%        filename = strcat('osafpc',num2str(img),'.jpg');
%        b=read(a,img);
%        c = rgb2gray(b);
%        imwrite(c,filename);
%    end

%  for x= 1:100
%   filename = strcat('osaf',num2str(x),'.jpg');
%   I = imread(filename);
%   EyeDetect = vision.CascadeObjectDetector('EyePairBig');
%   BB=step(EyeDetect,I);
%   CC=[0,0,0,0];
%   CC(1)=BB(1,1);
%   CC(2)=BB(1,2);
%   CC(3)=BB(1,3);
%   CC(4)=BB(1,4);
%   size(BB);
%   size(I);
%  %  imshow(I);
%  %  hold on
%  %  rectangle('Position',CC,'LineWidth',4,'LineStyle','-','EdgeColor','b');
%  %  title('Eyes Detection');
%    Eyes=imcrop(I,CC);
%   filename1 = strcat('osafEyes',num2str(x),'.jpg');
%   imwrite(Eyes,filename1);
%  end

data=zeros(300,2016);
labels=zeros(300,1);
myPathc= 'Training Data/ExtractedFeature(Eyes)/closed';
a=dir(fullfile(myPathc,'*.jpg'));
% fileNamesc={a.name};

myPatho= 'Training Data/ExtractedFeature(Eyes)/open';
b=dir(fullfile(myPatho,'*.jpg'));
% fileNameso={a.name};

myPathpco= 'Training Data/ExtractedFeature(Eyes)/partially closed:open';
c=dir(fullfile(myPathpco,'*.jpg'));
% fileNamespc0={a.name};

% P='Training Data/ExtractedFeatures(Eyes)/closed/';

% I = imread(strcat(myPathc,'/',a(1).name));
% I=imresize(I,[40 120]);
% figure,imshow(I);
% features = extractHOGFeatures(I);


for k = 1:100
    filename1 = [a(k).name];
    filename1 = strcat(myPathc,'/',filename1);
    I = imread(filename1);
    I=imresize(I,[40 120]);
    features = extractHOGFeatures(I);
    for z=1:2016
    data(k,z) = features(1,z);
    end
end

for k = 101:200
    filename2 = [b(k-100).name];
    filename2 = strcat(myPatho,'/',filename2);
    I = imread(filename2);
    I=imresize(I,[40 120]);
    features = extractHOGFeatures(I);
    for z=1:2016
    data(k,z) = features(1,z);
    end
end

for k = 201:300
    filename3 = [c(k-200).name];
    filename3 = strcat(myPathpco,'/',filename3);
    I = imread(filename3);
    I=imresize(I,[40 120]);
    features = extractHOGFeatures(I);
    for z=1:2016
    data(k,z) = features(1,z);
    end
end

for i= 1 :100
    labels(i,1)=0;
end
for j= 101 :200
    labels(j,1)=1;
end
for y= 201 :300
    labels(y,1)=2;
end

testdata = zeros(500,2016);

 t = VideoReader('test.MOV');
 num=t.NumberOfFrames;
 for var = 1:t.NumberOfFrames-1
%      filename = strcat('frame',num2str(img),'.jpg');
     b=read(t,var);
     c = rgb2gray(b);
     EyeDetect = vision.CascadeObjectDetector('EyePairBig');
     BB=step(EyeDetect,c);
     CC=[0,0,0,0];
     CC(1)=BB(1,1);
     CC(2)=BB(1,2);
     CC(3)=BB(1,3);
     CC(4)=BB(1,4);
%      size(BB);
%      size(I);
%      imshow(I);
%      hold on
%      rectangle('Position',CC,'LineWidth',4,'LineStyle','-','EdgeColor','b');
%      title('Eyes Detection');
     Eyes=imcrop(c,CC);
     d=imresize(Eyes,[40 120]);
%      filename = strcat('testframe',num2str(img),'.jpg');
     tfeatures = extractHOGFeatures(d);
     testdata(var , :) = tfeatures;
 end
 
Mdl = fitcknn(data,labels,'NumNeighbors',5,'Standardize',1);
% Mdl.ClassNames
% Mdl.Prior
outlabels = predict(Mdl,testdata);

blinks=zeros(500,1);
blinks(1,1)=outlabels(1,1);

j=1;
for i=2:t.NumberOfFrames-1
    if outlabels(i,1)~=blinks(j,1)
        blinks(j+1,1)=outlabels(i,1);
        j=j+1;
    end 
end

blinker = [1,2,0,2,1];
total = 0;
count =1;
for i=1:t.NumberOfFrames-1
    if blinks(i,1)== blinker(1,count)
        count = count + 1;
    end
          
    if count == 6
        count = 1;
        total = total + 1;
    end       
end


