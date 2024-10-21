package com.ding.dl4jpro.dl4jexperiments.demo;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.impl.graph.SparkComputationGraph;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Sgd;

import java.util.ArrayList;
import java.util.List;

/**https://mgubaidullin.github.io/deeplearning4j-docs/programmingguide/08_deploy
 * https://deeplearning4j.konduit.ai/multi-project/how-to-guides/developer-docs/github-actions-build-infra
 * ProjectName: deeplearning4jpro
 * ClassName: Demo01
 * Package: com.ding.dl4jpro.dl4jexperiments.demo
 * Description:
 *
 * @Author: ding
 * @Create 2024/10/14 14:48
 * @Version 1.0
 **/
public class Demo01 {
    private static final Object BATCH_SIZE = 100;

    /** https://blog.cloudera.com/deep-learning-on-apache-spark-and-hadoop-with-deeplearning4j/
     * val conf = new NeuralNetConfiguration.Builder()
     *  .seed(42)
     *  .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
     *  .iterations(1)
     *  .activation(Activation.SOFTMAX)
     *  .weightInit(WeightInit.XAVIER)
     *  .learningRate(0.01)
     *  .updater(Updater.NESTEROVS)
     *  .momentum(0.8)
     *  .graphBuilder()
     *  .addInputs("in")
     *  .addLayer("layer0",
     *    new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
     *      .activation(Activation.SOFTMAX)
     *      .nIn(4096)
     *      .nOut(257)
     *      .build(),
     *    "in")
     *  .setOutputs("layer0")
     *  .backprop(true)
     *  .build()
     * val model = new ComputationGraph(conf)
     * @param args
     */
    public static void main(String[] args) {

        SparkConf sparkConf = new SparkConf();
        JavaSparkContext sc = new JavaSparkContext(sparkConf);
//        new TrainedModelHelper(TrainedModels.VGG16);
//        new NeuralNetConfiguration.Builder();

        // 数据格式转换：将数据转换成模型输入的格式
        List<DataSet> trainDataList = new ArrayList<DataSet>();
        // 指定为自己的训练数据集
        DataSetIterator trainData = getTrainData();
        while (trainData.hasNext()){
            trainDataList.add(trainData.next());
        }
        JavaRDD<DataSet> JtrainData = sc.parallelize(trainDataList);

        // 训练主控配置
        // 设置批处理大小
        int BATCH_SIZE = 64;
        // 数据集对象大小
        int dataSetObjectSize = trainDataList.size();

        TrainingMaster tm = new ParameterAveragingTrainingMaster.Builder(dataSetObjectSize)
        .averagingFrequency(5)
                .workerPrefetchNumBatches(2)
                .batchSizePerWorker(BATCH_SIZE)
                .build();

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.RELU)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Sgd(0.05))
                // ... other hyperparameters
                .list().backpropType(BackpropType.Standard)
//                .backprop(true)
                .build();
        SparkDl4jMultiLayer sparkDl4jMultiLayer = new SparkDl4jMultiLayer(sc, conf, tm);
//        SparkComputationGraph sparkNet = new SparkComputationGraph(sc, conf, tm);

        sparkDl4jMultiLayer.fit(String.valueOf(JtrainData));

    }

    // 示例方法：你需要自己实现数据集的加载
    private static DataSetIterator getTrainData() {
        // 返回数据集迭代器（例如通过 CSV 或者数据库加载）
        // return new RecordReaderDataSetIterator(...);
        return null;  // 你需要实际的实现
    }
    // 示例方法：你需要自己实现模型的创建
    private static MultiLayerNetwork getModel() {
        // 返回已经定义的模型
        // return new MultiLayerNetwork(conf);
        return null;  // 你需要实际的实现
    }
}
