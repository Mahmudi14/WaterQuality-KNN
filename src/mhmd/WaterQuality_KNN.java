package mhmd;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Scanner;

public class WaterQuality_KNN {

    public static void main(String[] args) {
        double[][] dataDouble;
        double[][] data;
        ArrayList<double[]> dataArrayList = new ArrayList<>();
        File file = new File("src/dataset/waterQuality1.csv");

        //----------------------------------------------------------------------
        //Proses Baca data dari file
        //----------------------------------------------------------------------
        try {
            Scanner sc = new Scanner(file);
            sc.nextLine();//Skip line pertama
            while (sc.hasNextLine()) {
                String line = sc.nextLine();
                String[] arrLine = line.split(",");
                double[] arrBarisDouble = new double[arrLine.length];
                boolean add = true;
                for (int i = 0; i < arrLine.length; i++) {
                    String isi = arrLine[i];
                    if (isi.equalsIgnoreCase("#NUM!")) {
                        add = false;
                        break;
                    }
                    arrBarisDouble[i] = Double.parseDouble(isi);
                }
                if (add) {
                    dataArrayList.add(arrBarisDouble);
                }
            }
        } catch (FileNotFoundException ex) {
        }

        dataDouble = new double[dataArrayList.size()][];
        for (int i = 0; i < dataArrayList.size(); i++) {
            dataDouble[i] = dataArrayList.get(i);
        }

        //----------------------------------------------------------------------
        //Mencari Nilai MAX-MIN dari tiap kolom atribut
        //----------------------------------------------------------------------
        double[] min = new double[dataDouble[0].length];
        double[] max = new double[dataDouble[0].length];
        for (int i = 0; i < dataDouble[0].length; i++) {
            min[i] = Double.MAX_VALUE;
            max[i] = Double.MIN_VALUE;
        }

        for (int i = 0; i < dataDouble.length; i++) {
            for (int j = 0; j < dataDouble[i].length; j++) {
                double value = dataDouble[i][j];

                if (value < min[j]) {
                    min[j] = value;
                }
                if (value > max[j]) {
                    max[j] = value;
                }
            }
        }

        //----------------------------------------------------------------------
        //Proses Normalisasi MIN-MAX
        //----------------------------------------------------------------------
        data = new double[dataDouble.length][dataDouble[0].length];
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[i].length; j++) {
                double value = dataDouble[i][j];
                double normalValue = (value - min[j]) / (max[j] - min[j]);
                data[i][j] = normalValue;
            }
        }

        //----------------------------------------------------------------------
        //Proses Split Data
        //----------------------------------------------------------------------
        double[][] dataTraining;
        double[][] dataTesting;

        int numData = data.length;
        int percentDataTraining =80;
        int percentDataTesting = 20;
        double totalPercent = percentDataTraining + percentDataTesting;
        int numDataTraining = (int) Math.round((numData * (percentDataTraining / totalPercent)));
        int numDataTesting = (int) Math.round((numData * (percentDataTesting / totalPercent)));


//        int numDataTraining = 6000;
//        int numDataTesting = 1500;

        dataTraining = new double[numDataTraining][];
        dataTesting = new double[numDataTesting][];

        for (int i = 0; i < dataTraining.length; i++) {
            dataTraining[i] = data[i];
        }
        for (int i = 0; i < dataTesting.length; i++) {
            dataTesting[i] = data[i + numDataTraining];
        }

//        for (int i = numData; i > numData - numDataTesting; i--) {
//            dataTesting[numData - i] = data[i - 1];
//        }

        //----------------------------------------------------------------------
        //Proses KNN
        //----------------------------------------------------------------------
        int K = 9;
        double[][] result = new double[dataTesting.length][2];

        for (int i = 0; i < dataTesting.length; i++) {
            double[][] dataJarak = new double[dataTraining.length][3];
            double id = i;
            for (int j = 0; j < dataTraining.length; j++) {
                double sum = 0;
                for (int k = 0; k < dataTraining[i].length - 1; k++) {
                    double x2x1 = dataTraining[j][k] - dataTesting[i][k];
                    x2x1 = Math.pow(x2x1, 2);
                    sum += x2x1;
                }
                sum = Math.sqrt(sum);
                dataJarak[j][0] = j;
                dataJarak[j][1] = sum;
                dataJarak[j][2] = dataTraining[j][dataTraining[0].length - 1];
            }

            //Sorting
            for (int j = 0; j < dataJarak.length - 1; j++) {
                int i_MIN = j;
                double value = dataJarak[j][1];
                for (int k = j + 1; k < dataJarak.length; k++) {
                    if (value > dataJarak[k][1]) {
                        i_MIN = k;
                        value = dataJarak[k][1];
                    }
                }
                if (i_MIN != j) {
                    double temp_id = dataJarak[j][0];
                    double temp_value = dataJarak[j][1];
                    double temp_class = dataJarak[j][2];

                    dataJarak[j][0] = dataJarak[i_MIN][0];
                    dataJarak[j][1] = dataJarak[i_MIN][1];
                    dataJarak[j][2] = dataJarak[i_MIN][2];

                    dataJarak[i_MIN][0] = temp_id;
                    dataJarak[i_MIN][1] = temp_value;
                    dataJarak[i_MIN][2] = temp_class;
                }
            }

            //Mengambil data K tetangga terdekat
            double[][] k_dataJarak = new double[K][3];
            for (int j = 0; j < K; j++) {
                k_dataJarak[j][0] = dataJarak[j][0];
                k_dataJarak[j][1] = dataJarak[j][1];
                k_dataJarak[j][2] = dataJarak[j][2];
            }

            //Voting
            int safe = 0;
            int not_safe = 0;
            for (int j = 0; j < k_dataJarak.length; j++) {
                if (k_dataJarak[j][2] == 1.0) {
                    safe++;
                } else if (k_dataJarak[j][2] == 0.0) {
                    not_safe++;
                }
            }

            //Menentukan Predict class
            double predict = -1;
            if (safe > not_safe) {
                predict = 1.0;
            } else {
                predict = 0.0;
            }

            result[i][0] = id;
            result[i][1] = predict;
        }

        //----------------------------------------------------------------------
        //Evaluasi
        //----------------------------------------------------------------------
        double TP = 0;
        double FP = 0;
        double FN = 0;
        double TN = 0;

        for (int i = 0; i < dataTesting.length; i++) {
            double actualValue = dataTesting[i][dataTesting[0].length - 1];
            double predictValue = result[i][1];
            if (actualValue == 1.0 && predictValue == 1.0) {
                TP++;
            } else if (actualValue == 0.0 && predictValue == 1.0) {
                FP++;
            } else if (actualValue == 1.0 && predictValue == 0.0) {
                FN++;
            } else if (actualValue == 0.0 && predictValue == 00) {
                TN++;
            }
        }
        double akurasi = ((TP + TN) / (TP + FP + FN + TN));
        double precision = (TP / (TP + FP));
        double recall = (TP / (TP + FN));
        double F_Measure = (2 * ((precision * recall) / (precision + recall)));
        System.out.println("----------------------------------------------------");
        for (int i = 0; i < result.length; i++) {
            System.out.println(dataTesting[i][dataTesting[0].length - 1] + " - " + result[i][1]);
        }

        System.out.println("Banyak Data Training : " + dataTraining.length);
        System.out.println("Banyak Data Testing : " + dataTesting.length);
        System.out.println("+----------+");
        System.out.println("|"+TP+"\t"+FP+"|");
        System.out.println("|"+FN+"\t"+TN+"|");
        System.out.println("+----------+");
        System.out.println("Akurasi : " + (akurasi*100));
        System.out.println("Recall : " + (recall*100));
        System.out.println("Precision : " + (precision*100));
        System.out.println("F_Measure : " + (F_Measure*100));
    }
}































