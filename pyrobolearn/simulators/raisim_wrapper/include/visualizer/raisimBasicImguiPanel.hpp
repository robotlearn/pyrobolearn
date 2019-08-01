//
// Created by Jemin Hwangbo on 2/28/19.
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


#ifndef RAISIMOGREVISUALIZER_RAISIMBASICIMGUIPANEL_HPP
#define RAISIMOGREVISUALIZER_RAISIMBASICIMGUIPANEL_HPP
#include "guiState.hpp"

ImFont* fontBig;
ImFont* fontMid;
ImFont* fontSmall;

void imguiRenderCallBack() {


  ImGui::SetNextWindowPos({0, 0});
//  ImGui::SetNextWindowSize({400, 1000}, 0);
  if (!ImGui::Begin("RaiSim Application Window")) {
    // Early out if the window is collapsed, as an optimization.
    ImGui::End();
    return;
  }

  auto vis = raisim::OgreVis::get();
  auto world = vis->getWorld();
  vis->getPaused() = raisim::gui::manualStepping;

  unsigned long mask = 0;
  ImGui::PushFont(fontBig);
  ImGui::Text("Visualization");
  ImGui::Separator();
  ImGui::PopFont();

  ImGui::PushFont(fontMid);
  ImGui::Checkbox("Bodies", &raisim::gui::showBodies);
  ImGui::Checkbox("Collision Bodies", &raisim::gui::showCollision);
  ImGui::Checkbox("Contact Points", &raisim::gui::showContacts);
  ImGui::Checkbox("Contact Forces", &raisim::gui::showForces);
  ImGui::PopFont();

  if(raisim::gui::showBodies) mask |= raisim::OgreVis::RAISIM_OBJECT_GROUP;
  if(raisim::gui::showCollision) mask |= raisim::OgreVis::RAISIM_COLLISION_BODY_GROUP;
  if(raisim::gui::showContacts) mask |= raisim::OgreVis::RAISIM_CONTACT_POINT_GROUP;
  if(raisim::gui::showForces) mask |= raisim::OgreVis::RAISIM_CONTACT_FORCE_GROUP;

  vis->setVisibilityMask(mask);
  if(raisim::gui::manualStepping) {
    if(vis->getTakeNSteps() == -1)
      vis->getTakeNSteps() = 0;
  } else {
    vis->getTakeNSteps() = -1;
  }

  if (ImGui::CollapsingHeader("Simulation")) {
    ImGui::PushFont(fontMid);
    ImGui::Text("Sim time: %8.3f, Time step: %8.3f", world->getWorldTime(), world->getTimeStep());
    static int takeNSteps = 1;
    ImGui::Checkbox("Manual stepping", &raisim::gui::manualStepping);
    if(raisim::gui::manualStepping) {
      std::string tempString = "Remaining Steps: " + std::to_string(vis->getTakeNSteps());
      ImGui::Text("%s", tempString.c_str());
      ImGui::Text("Take "); ImGui::SameLine(); ImGui::InputInt("", &takeNSteps); ImGui::SameLine(); ImGui::Text(" steps"); ImGui::SameLine();
      if(ImGui::Button("Run"))
        vis->getTakeNSteps() += takeNSteps;
    } else {
      if(ImGui::Button("Set to real time"))
        vis->getRealTimeFactorReference() = 1.f;
      ImGui::SameLine();
      ImGui::SliderFloat("", &vis->getRealTimeFactorReference(), 1e-3, 1000, "Real time factor %5.4f", 10);
    }
    ImGui::PopFont();
  }

  auto selected = vis->getSelected();
  auto ro = std::get<0>(selected);
  auto li = std::get<1>(selected);

  if (ImGui::CollapsingHeader("Object data")) {
    if(ro) {
      ImGui::PushFont(fontBig);
      if(!ro->getName().empty()){
        ImGui::Text("%s", ("name: " + ro->getName() + "/" + vis->getSelectedGraphicalObject()->name).c_str());
      } else
        ImGui::Text("Unnamed object");

      raisim::Vec<3> pos; ro->getPosition_W(li, pos);
      raisim::Vec<3> vel; ro->getVelocity_W(li, vel);
      raisim::Vec<4> ori; raisim::Mat<3,3> mat; ro->getOrientation_W(li, mat); raisim::rotMatToQuat(mat, ori);
      ImGui::PopFont();

      ImGui::PushFont(fontMid);
      ImGui::Text("Position");
      ImGui::PopFont();

      ImGui::PushFont(fontMid);
      ImGui::Text("x = %2.2f, y = %2.2f, z = %2.2f", pos[0], pos[1], pos[2]);
      ImGui::PopFont();

      ImGui::PushFont(fontMid);
      ImGui::Text("Velocity");
      ImGui::PopFont();

      ImGui::PushFont(fontMid);
      ImGui::Text("x = %2.2f, y = %2.2f, z = %2.2f", vel[0], vel[1], vel[2]);
      ImGui::PopFont();

      ImGui::PushFont(fontMid);
      ImGui::Text("Orientation");
      ImGui::PopFont();

      ImGui::PushFont(fontMid);
      ImGui::Text("x = %2.2f, x = %2.2f, y = %2.2f, z = %2.2f", ori[0], ori[1], ori[2], ori[3]);
      ImGui::PopFont();

      ImGui::PushFont(fontMid);
      ImGui::Text("Ncontacts: %lu", ro->getContacts().size());
      ImGui::PopFont();
    }
  }

  if (ImGui::CollapsingHeader("Contacts")) {
    ImGui::PushFont(fontMid);
    ImGui::Text("Solver Iterations: %d", world->getContactSolver().getLoopCounter());
    ImGui::Text("Total number of contacts: %lu", world->getContactProblem()->size());
    std::vector<float> error;
    error.reserve(world->getContactSolver().getLoopCounter());

    for(int i=0; i<world->getContactSolver().getLoopCounter(); i++)
      error.push_back(float(log(world->getContactSolver().getErrorHistory()[i])));

    ImGui::PlotLines("Lines", &error[0], error.size(), 0, "avg 0.0", float(log(world->getContactSolver().getConfig().error_to_terminate)), 1.0f, ImVec2(500,300));

    const auto* problem = world->getContactProblem();

    for(int i=0; i<problem->size(); i++) {
      ImGui::Text("%i: Rank %i", i, problem->at(i).rank);
    }

    ImGui::PopFont();
  }

  if (ImGui::CollapsingHeader("Video recording")) {
    if(vis->isRecording()) {
      ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.8f, 0.2f, 1.f));
      ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.9f, 0.3f, 1.f));
      ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.5f, 0.9f, 0.5f, 1.f));

      if(ImGui::Button("Stop Recording ")) {
        RSINFO("Stop recording")
        raisim::OgreVis::get()->stopRecordingVideoAndSave();
      }

      ImGui::PopStyleColor(3);
    } else {
      ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.8f, 0.2f, 0.2f, 1.f));
      ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.9f, 0.3f, 0.3f, 1.f));
      ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.9f, 0.5f, 0.5f, 1.f));

      if(ImGui::Button("Record ")){
        RSINFO("Start recording")
        raisim::OgreVis::get()->startRecordingVideo(raisim::OgreVis::get()->getResourceDir() + "/test.mp4");
      }

      ImGui::PopStyleColor(3);
    }
  }

  if (ImGui::CollapsingHeader("Key maps")) {
    ImGui::Text("F1~4        : toggle visualization mask\n");
    ImGui::Text("Mouse L : orbital mode\n");
    ImGui::Text("Mouse R : free cam mode\n");
    ImGui::Text("Shift        : pan during free cam mode\n");
  }

  const float INDENT = ImGui::GetTreeNodeToLabelSpacing();
  if (ImGui::CollapsingHeader("Object List")) {
    ImGui::Indent(INDENT);
    auto& items = vis->getObjectSet();
    for(auto& it: items.set) {
      if(ImGui::TreeNode(it.first->getName().c_str())) {
        ImGui::PushFont(fontMid);
        ImGui::Indent(INDENT);
        if(ImGui::TreeNode("simulation objects")) {
          ImGui::Indent(INDENT);

          ImGui::Unindent(INDENT);
          ImGui::TreePop();
        }

        if(ImGui::TreeNode("visual objects")) {
          ImGui::Indent(INDENT);

          ImGui::Unindent(INDENT);
          ImGui::TreePop();
        }
        ImGui::Unindent(INDENT);
        ImGui::PopFont();
        ImGui::TreePop();
      }
    }
    ImGui::Unindent(INDENT);
  }

  ImGui::End();

}

void imguiSetupCallback() {

#define HI(v)   ImVec4(0.502f, 0.075f, 0.256f, v)
#define MED(v)  ImVec4(0.455f, 0.198f, 0.301f, v)
#define LOW(v)  ImVec4(0.232f, 0.201f, 0.271f, v)
  // backgrounds (@todo: complete with BG_MED, BG_LOW)
#define BG(v)   ImVec4(0.200f, 0.220f, 0.270f, v)
  // text
#define TEXT(v) ImVec4(0.860f, 0.930f, 0.890f, v)

  auto &style = ImGui::GetStyle();
  style.Alpha = 0.8;
  style.Colors[ImGuiCol_Text]                  = TEXT(0.78f);
  style.Colors[ImGuiCol_TextDisabled]          = TEXT(0.28f);
  style.Colors[ImGuiCol_WindowBg]              = ImVec4(0.13f, 0.14f, 0.17f, 1.00f);
  style.Colors[ImGuiCol_ChildWindowBg]         = BG( 0.58f);
  style.Colors[ImGuiCol_PopupBg]               = BG( 0.9f);
  style.Colors[ImGuiCol_Border]                = ImVec4(0.31f, 0.31f, 1.00f, 0.00f);
  style.Colors[ImGuiCol_BorderShadow]          = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
  style.Colors[ImGuiCol_FrameBg]               = BG( 1.00f);
  style.Colors[ImGuiCol_FrameBgHovered]        = MED( 0.78f);
  style.Colors[ImGuiCol_FrameBgActive]         = MED( 1.00f);
  style.Colors[ImGuiCol_TitleBg]               = LOW( 1.00f);
  style.Colors[ImGuiCol_TitleBgActive]         = HI( 1.00f);
  style.Colors[ImGuiCol_TitleBgCollapsed]      = BG( 0.75f);
  style.Colors[ImGuiCol_MenuBarBg]             = BG( 0.47f);
  style.Colors[ImGuiCol_ScrollbarBg]           = BG( 1.00f);
  style.Colors[ImGuiCol_ScrollbarGrab]         = ImVec4(0.09f, 0.15f, 0.16f, 1.00f);
  style.Colors[ImGuiCol_ScrollbarGrabHovered]  = MED( 0.78f);
  style.Colors[ImGuiCol_ScrollbarGrabActive]   = MED( 1.00f);
  style.Colors[ImGuiCol_CheckMark]             = ImVec4(0.71f, 0.22f, 0.27f, 1.00f);
  style.Colors[ImGuiCol_SliderGrab]            = ImVec4(0.47f, 0.77f, 0.83f, 0.14f);
  style.Colors[ImGuiCol_SliderGrabActive]      = ImVec4(0.71f, 0.22f, 0.27f, 1.00f);
  style.Colors[ImGuiCol_Button]                = ImVec4(0.47f, 0.77f, 0.83f, 0.14f);
  style.Colors[ImGuiCol_ButtonHovered]         = MED( 0.86f);
  style.Colors[ImGuiCol_ButtonActive]          = MED( 1.00f);
  style.Colors[ImGuiCol_Header]                = MED( 0.76f);
  style.Colors[ImGuiCol_HeaderHovered]         = MED( 0.86f);
  style.Colors[ImGuiCol_HeaderActive]          = HI( 1.00f);
  style.Colors[ImGuiCol_Column]                = ImVec4(0.14f, 0.16f, 0.19f, 1.00f);
  style.Colors[ImGuiCol_ColumnHovered]         = MED( 0.78f);
  style.Colors[ImGuiCol_ColumnActive]          = MED( 1.00f);
  style.Colors[ImGuiCol_ResizeGrip]            = ImVec4(0.47f, 0.77f, 0.83f, 0.04f);
  style.Colors[ImGuiCol_ResizeGripHovered]     = MED( 0.78f);
  style.Colors[ImGuiCol_ResizeGripActive]      = MED( 1.00f);
  style.Colors[ImGuiCol_PlotLines]             = TEXT(0.63f);
  style.Colors[ImGuiCol_PlotLinesHovered]      = MED( 1.00f);
  style.Colors[ImGuiCol_PlotHistogram]         = TEXT(0.63f);
  style.Colors[ImGuiCol_PlotHistogramHovered]  = MED( 1.00f);
  style.Colors[ImGuiCol_TextSelectedBg]        = MED( 0.43f);
  // [...]
  style.Colors[ImGuiCol_ModalWindowDarkening]  = BG( 0.73f);

  style.WindowPadding            = ImVec2(6, 4);
  style.WindowRounding           = 0.0f;
  style.FramePadding             = ImVec2(5, 2);
  style.FrameRounding            = 3.0f;
  style.ItemSpacing              = ImVec2(7, 1);
  style.ItemInnerSpacing         = ImVec2(1, 1);
  style.TouchExtraPadding        = ImVec2(0, 0);
  style.IndentSpacing            = 6.0f;
  style.ScrollbarSize            = 12.0f;
  style.ScrollbarRounding        = 16.0f;
  style.GrabMinSize              = 20.0f;
  style.GrabRounding             = 2.0f;

  style.WindowTitleAlign.x = 0.50f;

  style.Colors[ImGuiCol_Border] = ImVec4(0.539f, 0.479f, 0.255f, 0.162f);
  style.FrameBorderSize = 0.0f;
  style.WindowBorderSize = 1.0f;

  ImGuiIO &io = ImGui::GetIO();
  fontBig = io.Fonts->AddFontFromFileTTF((raisim::OgreVis::get()->getResourceDir() + "/font/DroidSans.ttf").c_str(), 25.0f);
  fontMid = io.Fonts->AddFontFromFileTTF((raisim::OgreVis::get()->getResourceDir() + "/font/DroidSans.ttf").c_str(), 22.0f);
  fontSmall = io.Fonts->AddFontFromFileTTF((raisim::OgreVis::get()->getResourceDir() + "/font/DroidSans.ttf").c_str(), 16.0f);
}

#endif //RAISIMOGREVISUALIZER_RAISIMBASICIMGUIPANEL_HPP
