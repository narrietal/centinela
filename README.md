# Project Centinela

![Fin detection](images/example_fin_detection.jpg)

## Introduction
Project Centinela is a collaboration among local environmental activists, researchers, and engineers aimed at **raising awareness of and protecting the Short-Finned Pilot Whales** in the Canary Islands, Spain. The islands are experiencing **unsustainable practices** due to massive tourism attractions, which not only impact the locals but also degrade the unique wildlife of the area. Water and noise pollution, along with deadly animal-ship encounters, are among the many direct causes of such mispractices. Among all the species, the Short-Finned Pilot Whales (Calderones Tropicales in Spanish) are of particular interest, as there is a group of **around 100 individuals that reside in the islands year-round**. This phenomenon has only been documented in **four other places in the world**. Little is known of these cetaceans and in 2008 it was officially listed on the IUCN Red List as Data Deficient.

This project aims to raise awareness of and protect such a unique species by creating an identification algorithm for the resident individuals with the aim of accomplishing two goals: gathering data for research about the species and providing tourists and locals with means to become more familiar with their cetacean neighbors.

This repository hosts a website with a demo of the recognition and identification algorithm based on the identification of their dorsal fin:

[Centinela demo](https://narrietal.github.io/centinela/)

1. A first model for detecting dorsal fins in images.
2. A second model for individual identification.

The models have been trained using a collection of images of Tropical Whales containing more than 1600 images and a total of 20 individuals.

The dorsal fin detection model has an average precision (MAP50) of **98.9%**.
The model for identifying the 20 individuals has an accuracy of **95.6%**.

## Content

The prototype consists of three notebooks:

* Centinela_project_fin_detection: Training and testing of fin detection model
* Centinela_project_identification: Training and testing of fin identification model
* Centinela_project_prototype: Inference algorithm as proof of concept

Author:
- Name: Nicolas Arrieta Larraza
- Contact: n.arrieta.larraza@gmail.com
  
