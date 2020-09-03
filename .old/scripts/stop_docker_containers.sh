#!/bin/bash

# @author:    Patrik Purgai
# @copyright: Copyright 2019, xlmr-hungarian
# @license:   MIT
# @email:     purgai.patrik@gmail.com
# @date:      2019.07.12.


for container in `sudo docker ps -a -q`;
do
    echo "Stopping container - $container";
    sudo docker stop $container;
done
