%% import data
clc;clear;
load dataClean;
input=dataClean(:,1:11)';
target=dataClean(:,1)';
results=zeros(1,21);
mseTrain=zeros(1,10);
mseTest=zeros(1,10);
for i=10:20
    %% define net
    hiddenLayerSize=i;
    trainfcn='trainbr';
    net=feedforwardnet(hiddenLayerSize,trainfcn);
    net.input.processFcns = {'removeconstantrows','mapminmax'};
    net.output.processFcns = {'removeconstantrows','mapminmax'};
    %% set parameters
    net.divideFcn='divideind';
    net.divideParam.trainInd=(1:200);
    net.divideParam.valInd=(200:220);
    net.divideParam.testInd=(221:242);
    net.trainParam.epochs=2000;
    net.trainParam.lr=0.005;
    net.trainParam.goal=0;
    net.trainParam.mc=0.9;
    net.trainParam.max_fail=10;
    %% train
    a0=0;
    b0=0;
    for j=1:5
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
        a0=a0+trainPerformance;
        b0=b0+testPerformance;
    end
    mseTrain(i-9)=a0/5;
    mseTest(i-9)=b0/15;
end

    