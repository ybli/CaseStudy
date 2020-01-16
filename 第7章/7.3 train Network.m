%% import data
clc;clear;
load dataClean;
input=dataClean(:,1:11)';
target=dataClean(:,1)';

%% define net
hiddenLayerSize=16;
trainfcn='trainbr';
net=feedforwardnet(hiddenLayerSize,trainfcn);
net.input.processFcns = {'removeconstantrows','mapminmax'};
net.output.processFcns = {'removeconstantrows','mapminmax'};

%% set parameters
net.divideFcn='divideind';
net.divideParam.trainInd=(1:200);
net.divideParam.valInd=(201:220);
net.divideParam.testInd=(221:242);
net.trainParam.epochs=10000;
net.trainParam.lr=0.005;
net.trainParam.goal=0;
net.trainParam.mc=0.9;
net.trainParam.max_fail=10;
%% train
[net,tr]=train(net,input,target);

%% analyze results
outputs=net(input);
trainOutputs=outputs(tr.trainInd);
trainTargets=target(tr.trainInd);
testOutputs=outputs(tr.testInd);
testTargets=target(tr.testInd);
trainPerformance=perform(net,trainTargets,trainOutputs);
testPerformance=perform(net,testTargets,testOutputs);
testErrors=gsubtract(testTargets,testOutputs);
testErrorRatio=abs(testErrors)./testTargets.*100;
testERAverage=sum(testErrorRatio)/22;