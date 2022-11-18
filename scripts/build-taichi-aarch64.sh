#!/bin/bash

pushd framework/android

echo > local.properties
echo "sdk.dir=$ANDROID_HOME" >> local.properties

java "-Dorg.gradle.appname=taichi_aot_demo" \
    "-Dhttps.proxyHost=127.0.0.1" \
    "-Dhttps.proxyPort=1080" \
    "-Dhttp.nonProxyHosts=*.nonproxyrepos.com|localhost" \
    -classpath "gradle/wrapper/gradle-wrapper.jar" \
    org.gradle.wrapper.GradleWrapperMain build

popd
