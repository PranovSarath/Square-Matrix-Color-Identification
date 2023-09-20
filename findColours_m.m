function res=findColours_m(filename,varargin)
%my dummy find colours, all it does is read the actual answer file and
%maybe flip it

% get solution file
if length(varargin)==1
    mat_filename = varargin{1};
else
    [folder, baseFileName, ~] = fileparts(filename);
    mat_filename = fullfile(folder, sprintf('%s.mat',baseFileName));
end
fprintf('Loading %s\n', mat_filename)
load(mat_filename,'res')

%random rotation
k = randi(4)-1;
res = rot90(res,k);
if randi(2)==1
    res=flipud(res);
end






