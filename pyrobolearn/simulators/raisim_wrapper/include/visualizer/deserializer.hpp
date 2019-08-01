//
// Created by Jemin Hwangbo on 3/3/19.
// MIT License
//
// Copyright (c) 2019-2019 Robotic Systems Lab, ETH Zurich
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef RAISIMOGREVISUALIZER_DESERIALIZER_HPP
#define RAISIMOGREVISUALIZER_DESERIALIZER_HPP
#include "raisim/OgreVis.hpp"
#include "raisim/RaisimServer.hpp"
#include <experimental/filesystem>

using namespace raisim;

namespace raisim {

class Deserializer {

 public:
  explicit Deserializer(const std::string &resDir) {
    resDir_ = resDir;
    if(resDir_.back() =='/')
      resDir_ = resDir.substr(0, resDir_.size()-1);

    receiveVector_.resize(raisim::RaisimServer::SEND_BUFFER_SIZE);
  };

  void estabilishConnection() {
    ///
    sock_ = 0;

    std::vector<char> buffer;
    buffer.resize(raisim::RaisimServer::SEND_BUFFER_SIZE);

    RSFATAL_IF((sock_ = socket(AF_INET, SOCK_STREAM, 0)) < 0, "Socket creation error");

    memset(&serv_addr, '0', sizeof(serv_addr));

    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(raisim::RaisimServer::RAISIM_PORT);

    // Convert IPv4 and IPv6 addresses from text to binary form
    RSFATAL_IF(inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr) <= 0, "Invalid address");
//    RSFATAL_IF(connect(sock_, (sockaddr *) &serv_addr, sizeof(serv_addr)) < 0, "Connection failed");

    RSINFO("Waiting for a response from a server")

    while(connect(sock_, (sockaddr *) &serv_addr, sizeof(serv_addr)) < 0) {

    }
  }

  int connectToServer() {
    return connect(sock_, (sockaddr *) &serv_addr, sizeof(serv_addr));
  }
  
  void closeSocket() {
    close(sock_);
  }

  /// return if reinitialization is necessary
  int updatePosition() {
    int request = raisim::RaisimServer::ClientMessageType::REQUEST_OBJECT_POSITION;
    send(sock_, &request, sizeof(int), 0);

    auto status = readData();
    if(!status)
      return status;

    auto vis = OgreVis::get();

    char* data = &receiveVector_[0];

    int state;
    data = RaisimServer::get(data, &state);

    if(state == RaisimServer::STATUS_TERMINATING)
      return 0;

    RaisimServer::ServerMessageType type;
    data = RaisimServer::get(data, &type);

    if(type == RaisimServer::ServerMessageType::NO_MESSAGE) {
      usleep(5e4);
      return 1;
    }

    unsigned long configurationNumber;
    data = RaisimServer::get(data, &configurationNumber);

    size_t numberOfObjects;
    data = RaisimServer::get(data, &numberOfObjects);

    for(size_t i=0; i<numberOfObjects; i++) {
      // set name length
      size_t localIdxSize;
      data = RaisimServer::get(data, &localIdxSize);

      for(size_t j=0; j < localIdxSize; j++) {
        std::string name;
        data = RaisimServer::getString(data, name);

        double posX, posY, posZ;
        double quatW, quatx, quaty, quatz;

        data = RaisimServer::get(data, &posX);
        data = RaisimServer::get(data, &posY);
        data = RaisimServer::get(data, &posZ);

        data = RaisimServer::get(data, &quatW);
        data = RaisimServer::get(data, &quatx);
        data = RaisimServer::get(data, &quaty);
        data = RaisimServer::get(data, &quatz);

        vis->getVisualObjectList()[name].graphics->setPosition(posX, posY, posZ);
        vis->getVisualObjectList()[name].graphics->setOrientation(quatW, quatx, quaty, quatz);
      }
    }
    memset(&receiveVector_[0], 0, receiveVector_.size() * sizeof(receiveVector_[0]));
    return status;
  }

  inline int init() {
    int request = raisim::RaisimServer::ClientMessageType::REQUEST_INITIALIZATION;
    send(sock_, &request, sizeof(int), 0);
    if(!readData())
      return 0;

    char* data = &receiveVector_[0];

    auto vis = OgreVis::get();
    vis->clearVisualObject();

    int state;
    data = RaisimServer::get(data, &state);

    if(state == RaisimServer::STATUS_TERMINATING)
      return 0;

    RaisimServer::ServerMessageType messageType;
    data = RaisimServer::get(data, &messageType);

    data = RaisimServer::get(data, &configurationNumber_);

    size_t size;
    auto &vobVec = vis->getVisualObjectList();
    data = RaisimServer::get(data, &size);
    std::vector<float> heights;

    for (size_t i = 0; i < size; i++) {
      std::string meshName;
      size_t obIndex;
      data = RaisimServer::get(data, &obIndex);
      raisim::Vec<3> scale, offset = {0, 0, 0};
      raisim::Mat<3, 3> rot;
      rot.setIdentity();
      raisim::ObjectType type;
      data = RaisimServer::get(data, &type);
      float radius, height, x, y, z;

      switch (type) {
        case raisim::ObjectType::SPHERE:
          data = RaisimServer::get(data, &radius);
          scale = {radius, radius, radius};
          meshName = "sphereMesh";
          vis->addVisualObject(std::to_string(obIndex), meshName, "default", scale);
          break;

        case raisim::ObjectType::BOX:
          data = RaisimServer::get(data, &x);
          data = RaisimServer::get(data, &y);
          data = RaisimServer::get(data, &z);
          scale = {x, y, z};
          meshName = "cubeMesh";
          vis->addVisualObject(std::to_string(obIndex), meshName, "default", scale);
          break;

        case raisim::ObjectType::CYLINDER:
          data = RaisimServer::get(data, &radius);
          data = RaisimServer::get(data, &height);
          scale = {radius, radius, height};
          meshName = "cylinderMesh";
          vis->addVisualObject(std::to_string(obIndex), meshName, "default", scale);
          break;

        case raisim::ObjectType::CAPSULE:
          data = RaisimServer::get(data, &radius);
          data = RaisimServer::get(data, &height);
          scale = {radius, radius, height};
          meshName = "capsuleMesh";
          break;

        case raisim::ObjectType::HALFSPACE:
          data = RaisimServer::get(data, &height);
          scale = {20, 20, 1};
          offset = {0, 0, height};
          meshName = "planeMesh";
          vis->addVisualObject(std::to_string(obIndex), meshName, "checkerboard_green", scale, false, 1<<0);
          break;

        case raisim::ObjectType::HEIGHTMAP:
          float centX, centY, sizeX, sizeY;
          size_t sampleX, sampleY, heightSize;

          data = RaisimServer::get(data, &centX);
          data = RaisimServer::get(data, &centY);
          data = RaisimServer::get(data, &sizeX);
          data = RaisimServer::get(data, &sizeY);
          data = RaisimServer::get(data, &sampleX);
          data = RaisimServer::get(data, &sampleY);
          data = RaisimServer::get(data, &heightSize);
          heights.resize(heightSize);
          data = RaisimServer::getN(data, heights.data(), heightSize);
          vis->buildHeightMap(std::to_string(obIndex), sampleX, sizeX, centX, sampleY, sizeY, centY, heights);

          scale = {1, 1, 1};
          offset = {0, 0, height};
          meshName = std::to_string(obIndex);
          vis->addVisualObject(std::to_string(obIndex), meshName, "default", scale);
          break;

        case raisim::ObjectType::ARTICULATED_SYSTEM:
          std::string objResDir, topDir, localObjResDir;
          data = RaisimServer::getString(data, objResDir);
          topDir = raisim::getFileName(objResDir);
          localObjResDir = resDir_ + separator() + topDir;

          RSFATAL_IF(!raisim::directoryExists(localObjResDir),
                     "Required resource directory " + localObjResDir + " is missing")

          for (size_t visItem = 0; visItem < 2; visItem++) {
            size_t numberOfVisObjects;
            data = RaisimServer::get(data, &numberOfVisObjects);

            for (size_t j = 0; j < numberOfVisObjects; j++) {
              raisim::Shape::Type shape;
              data = RaisimServer::get(data, &shape);
              unsigned long int group;
              data = RaisimServer::get(data, &group);

              std::string subName = std::to_string(obIndex) + separator() + std::to_string(visItem) + separator() + std::to_string(j);

              if (shape == Shape::Mesh) {
                std::string meshFile, fileName;
                data = RaisimServer::getString(data, meshFile);
                double sx, sy, sz;
                data = RaisimServer::get(data, &sx);
                data = RaisimServer::get(data, &sy);
                data = RaisimServer::get(data, &sz);
                scale = {sx, sy, sz};


                fileName = raisim::getFileName(meshFile);
                vis->loadMeshFile(resDir_ + separator() + topDir + separator() + fileName,
                                  topDir + separator() + fileName);
                vis->addVisualObject(subName,
                                     topDir + separator() + fileName,
                                     "default",
                                     scale,
                                     true,
                                     1<<group);
              } else {
                std::vector<double> visParam;
                data = RaisimServer::getStdVector(data, visParam);
                switch (shape) {
                  case Shape::Box:
                    scale = {visParam[0], visParam[1], visParam[2]};
                    meshName = "cubeMesh";
                    break;

                  case Shape::Capsule:
                    scale = {visParam[0], visParam[0], visParam[1]};
                    meshName = "capsuleMesh";
                    break;

                  case Shape::Cylinder:
                    scale = {visParam[0], visParam[0], visParam[1]};
                    meshName = "cylinderMesh";
                    break;

                  case Shape::Sphere:
                    scale = {visParam[0], visParam[0], visParam[0]};
                    meshName = "sphereMesh";
                    break;
                }
                vis->addVisualObject(subName, meshName, "default", scale, true, 1<<group);
                vis->getVisualObjectList()[subName].graphics->setScale(scale[0], scale[1], scale[2]);
              }
            }
          }
          break;
      }
    }
    memset(&receiveVector_[0], 0, receiveVector_.size() * sizeof(receiveVector_[0]));

    return 1;
  }

 private:


  int readData() {
    char *data = &receiveVector_[0];
    char footer = 'c';

    while (footer == 'c') {
      valread = recv(sock_, data, raisim::RaisimServer::MAXIMUM_PACKET_SIZE, 0);
      if(valread == 0) break;
      footer = data[raisim::RaisimServer::MAXIMUM_PACKET_SIZE - raisim::RaisimServer::FOOTER_SIZE];
      data += valread - raisim::RaisimServer::FOOTER_SIZE;
    }

    return data - &receiveVector_[0];
  }

  int sock_ = 0, valread;
  std::vector<char> receiveVector_;
  sockaddr_in serv_addr;
  unsigned long configurationNumber_;

  std::string resDir_;
};

}

#endif //RAISIMOGREVISUALIZER_DESERIALIZER_HPP
