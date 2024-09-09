package org.example;

public class Main {
    public static void main(String[] args) {
        MilvusSDKWrapper.create();
        MilvusSDKWrapper.test(1);
        System.out.println("Finished");
    }
}