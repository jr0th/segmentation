#!/bin/bash
rm -r training validation test
mkdir training validation test

mkdir training/x_big
mkdir training/y_big_label_binary_2 training/y_big_label_binary_4 training/y_big_label_binary_6 training/y_big_label_binary_8
mkdir training/y_big_boundary_2 training/y_big_boundary_4 training/y_big_boundary_6 training/y_big_boundary_8
mkdir training/y_big_label_single_channel
mkdir training/y_big_boundary_soft

mkdir training/x training/x/all
mkdir training/y_label_binary_2 training/y_label_binary_4 training/y_label_binary_6 training/y_label_binary_8
mkdir training/y_label_binary_2/all training/y_label_binary_4/all training/y_label_binary_6/all training/y_label_binary_8/all
mkdir training/y_boundary_2 training/y_boundary_4 training/y_boundary_6 training/y_boundary_8
mkdir training/y_boundary_2/all training/y_boundary_4/all training/y_boundary_6/all training/y_boundary_8/all
mkdir training/y_label_single_channel training/y_label_single_channel/all
mkdir training/y_boundary_soft training/y_boundary_soft/all

mkdir validation/x_big
mkdir validation/y_big_label_binary_2 validation/y_big_label_binary_4 validation/y_big_label_binary_6 validation/y_big_label_binary_8
mkdir validation/y_big_boundary_2 validation/y_big_boundary_4 validation/y_big_boundary_6 validation/y_big_boundary_8
mkdir validation/y_big_label_single_channel
mkdir validation/y_big_boundary_soft

mkdir validation/x validation/x/all
mkdir validation/y_label_binary_2 validation/y_label_binary_4 validation/y_label_binary_6 validation/y_label_binary_8
mkdir validation/y_label_binary_2/all validation/y_label_binary_4/all validation/y_label_binary_6/all validation/y_label_binary_8/all
mkdir validation/y_boundary_2 validation/y_boundary_4 validation/y_boundary_6 validation/y_boundary_8
mkdir validation/y_boundary_2/all validation/y_boundary_4/all validation/y_boundary_6/all validation/y_boundary_8/all
mkdir validation/y_label_single_channel validation/y_label_single_channel/all
mkdir validation/y_boundary_soft validation/y_boundary_soft/all

mkdir test/x_big
mkdir test/y_big_label_binary_2 test/y_big_label_binary_4 test/y_big_label_binary_6 test/y_big_label_binary_8
mkdir test/y_big_boundary_2 test/y_big_boundary_4 test/y_big_boundary_6 test/y_big_boundary_8
mkdir test/y_big_label_single_channel
mkdir test/y_big_boundary_soft

mkdir test/x test/x/all
mkdir test/y_label_binary_2 test/y_label_binary_4 test/y_label_binary_6 test/y_label_binary_8
mkdir test/y_label_binary_2/all test/y_label_binary_4/all test/y_label_binary_6/all test/y_label_binary_8/all
mkdir test/y_boundary_2 test/y_boundary_4 test/y_boundary_6 test/y_boundary_8
mkdir test/y_boundary_2/all test/y_boundary_4/all test/y_boundary_6/all test/y_boundary_8/all
mkdir test/y_label_single_channel test/y_label_single_channel/all
mkdir test/y_boundary_soft test/y_boundary_soft/all
