function Helios_fcn(train_imds, train_pxds,...
                                test_imds,...
                                test_pxds,...
                                train_flag)
                            
% --- Training --- %


if (train_flag)
    imds            = train_imds;
    pxds            = train_pxds;
    imageSize       = [360 720];
    numClasses      = 3;
    lgraph          = fcnLayers(imageSize,numClasses);
    save('lgraph','lgraph');
    pximds          = pixelLabelImageDatastore(imds,pxds);
    options         = trainingOptions(  'sgdm','InitialLearnRate',1e-3, ...
                                        'VerboseFrequency',10,...
                                        'MiniBatchSize',7,...
                                        'MaxEpochs',50,...
                                        'ExecutionEnvironment','multi-gpu');
    diary diary.txt;
    disp(strcat("=========",string(datetime('now')),"=============="));
    tic
        trained_net = trainNetwork(pximds,lgraph,options);
    toc
    disp("===============================================")
    diary off;
    save('trained_net','trained_net');
end





% --- Running trained FCN on training set--- %
gpuDevice(1); % Reset GPU
if not(train_flag)
    load('trained_net.mat');
end
imds          = train_imds;
pxdsTruth     = train_pxds;
num_test      = size(train_imds.Files(),1);
for i = 1:num_test
     I             = imread(imds.Files{i});
     C             = semanticseg(I,trained_net);
     I_bin         = 255*(C == 'c');
     [filepath,name,ext] = fileparts(imds.Files{i});
     imwrite(I_bin, "segmented_images/training/"+name+".png");    
end
pxdsResults = semanticseg(imds,trained_net, ...
    'MiniBatchSize',5, ...
    'WriteLocation',tempdir, ...
    'Verbose',false);
metrics_training = evaluateSemanticSegmentation(pxdsResults,pxdsTruth,'Verbose',false);
display("========== Training Set Start=========");
display(metrics_training.DataSetMetrics);
display(metrics_training.ClassMetrics);
display("========== Training Set End =========");

 % --- Running trained FCN on Testing set--- %
gpuDevice(1); % Reset GPU
if not(train_flag)
    load('trained_net.mat');
end
imds          = test_imds;
pxdsTruth     = test_pxds;
num_test      = size(test_imds.Files(),1);
for i = 1:num_test
     I             = imread(imds.Files{i});
     C             = semanticseg(I,trained_net);
     I_bin         = 255*(C == 'c');
     [filepath,name,ext] = fileparts(imds.Files{i});
     imwrite(I_bin, "segmented_images/testing/"+name+".png");    
end
pxdsResults = semanticseg(imds,trained_net, ...
    'MiniBatchSize',5, ...
    'WriteLocation',tempdir, ...
    'Verbose',false);
metrics_testing = evaluateSemanticSegmentation(pxdsResults,pxdsTruth,'Verbose',false);
display("========== Testing Set Start=========");
display(metrics_testing.DataSetMetrics);
display(metrics_testing.ClassMetrics);
display("========== Testing Set End =========");
end