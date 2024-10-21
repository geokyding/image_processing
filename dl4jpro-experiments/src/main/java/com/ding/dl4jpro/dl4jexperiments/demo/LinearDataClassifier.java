package com.ding.dl4jpro.dl4jexperiments.demo;

import org.apache.commons.io.FilenameUtils;
import org.datavec.api.split.FileSplit;
import org.nd4j.common.resources.Downloader;

import java.io.File;
import java.net.URL;

/**
 * ProjectName: deeplearning4jpro
 * ClassName: LinearDataClassifier
 * Package: com.ding.dl4jpro.dl4jexperiments.demo
 * Description:
 *
 * @Author: ding
 * @Create 2024/10/18 15:21
 * @Version 1.0
 **/
@SuppressWarnings("DuplicatedCode")
public class LinearDataClassifier {
    private static boolean visualize = true;
    private static String dataLocalPath;
    public static void main(String[] args) throws Exception {
        int seed = 1024;
        double learningRate = 0.001;
        int batch = 50;
        int nEpochs = 30;
        int inputs = 2;
        int outputs = 2;
        int numHiddenNodes = 20;
        String DATA_FOLDER = "dl4j-examples";
        String ZIP_FILE = "DataExamples.zip";
        String dataUrl = "https://dl4jdata.blob.core.windows.net/dl4j-examples/classification.zip";
        String tempdir = System.getProperty("java.io.tmpdir");
        String downloadPath = FilenameUtils.concat("D:\\works\\codetest\\dl4jpro\\datadir", ZIP_FILE);

        String extractDir = FilenameUtils.concat("D:\\works\\codetest\\dl4jpro\\datadir", "dl4j-examples-data/" + DATA_FOLDER);
        int downloadRetries = 10;
        Downloader.downloadAndExtract("classFiles",new URL(dataUrl),
                new File(extractDir),
                new File(downloadPath),
                "dba31e5838fe15993579edbf1c60c355",
                downloadRetries);

        // 数据准备：使用FileSplit和RecordReader对数据进行矢量化为DataVec数据形式
        /*https://mgubaidullin.github.io/deeplearning4j-docs/cn/datavec
        * https://zhuanlan.zhihu.com/p/60119869
        * DL4J有自己的特殊的数据结构DataVec，所有的输入数据在进入神经网络之前要先经过向量化。向量化后的结果就是一个行数不限的单列矩阵。
        * 熟悉Hadoop/MapReduce的朋友肯定知道它的输入用InputFormat来确定具体的InputSplit和RecordReader。DataVec也有自己FileSplit和RecordReader，
        * 并且对于不同的数据类型（文本、CSV、音频、图像、视频等），有不同的RecordReader，下面是一个图像的例子。
        * */
        FileSplit fileSplit = new FileSplit(new File(""));

        System.out.println("");


    }

}
