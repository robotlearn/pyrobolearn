using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public enum VibrationForce
{
    Light,
    Medium,
    Hard,
}


public class OculusHaptics : MonoBehaviour
{
    OVRInput.Controller controllerMask;

    private OVRHapticsClip clipLight;
    private OVRHapticsClip clipMedium;
    private OVRHapticsClip clipHard;

    public float lowViveHaptics { get; private set; }
    public float mediumViveHaptics { get; private set; }
    public float hardViveHaptics { get; private set; }

    private byte[] noize;


    private void Start()
    {
        InitializeOVRHaptics();
        //byte[] noize = { 250 };
        //clipHard = new OVRHapticsClip(noize, 1);
    }

    private void InitializeOVRHaptics()
    {

        //int cnt = 10;
        //clipLight = new OVRHapticsClip(cnt);
        //clipMedium = new OVRHapticsClip(cnt);
        ////clipHard = new OVRHapticsClip(cnt);
        //for (int i = 0; i < cnt; i++)
        //{
        //    clipLight.Samples[i] = i % 2 == 0 ? (byte)0 : (byte)45;
        //    clipMedium.Samples[i] = i % 2 == 0 ? (byte)0 : (byte)100;
        //    //clipHard.Samples[i] = i % 2 == 0 ? (byte)0 : (byte)180;
        //}

        //clipLight = new OVRHapticsClip(clipLight.Samples, clipLight.Samples.Length);
        //clipMedium = new OVRHapticsClip(clipMedium.Samples, clipMedium.Samples.Length);
        //clipHard = new OVRHapticsClip(clipHard.Samples, clipHard.Samples.Length);
    }


    void OnEnable()
    {
        InitializeOVRHaptics();
    }

    public void Vibrate(VibrationForce vibrationForce)
    {
        //var channel = OVRHaptics.RightChannel;
        //if (controllerMask == OVRInput.Controller.LTouch)
        //    channel = OVRHaptics.LeftChannel;

        //channel = OVRHaptics.Channels[1];

        //switch (vibrationForce)
        //{
        //    case VibrationForce.Light:
        //        channel.Preempt(clipLight);
        //        break;
        //    case VibrationForce.Medium:
        //        channel.Preempt(clipMedium);
        //        break;
        //    case VibrationForce.Hard:
        //        channel.Preempt(clipHard);
        //        break;
        //}


        OVRHaptics.Channels[0].Clear();
        OVRHaptics.Channels[1].Clear();

        if (vibrationForce == VibrationForce.Light)
            noize = new byte[10] { 60, 60, 60, 60, 60, 60, 60, 60, 60, 60 };
        else if (vibrationForce == VibrationForce.Medium)
            noize = new byte[10] { 120, 120, 120 , 120, 120, 120, 120, 120, 120, 120 };
        else
            noize = new byte[10] { 240, 240, 240, 240, 240, 240, 240, 240, 240, 240 };

        OVRHaptics.Channels[1].Preempt(new OVRHapticsClip(noize, 10));
        OVRHaptics.Process();
    }

    public IEnumerator VibrateTime(VibrationForce force, float time)
    {
        //bool forcedHaptic = true;
        var channel = OVRHaptics.RightChannel;
        if (controllerMask == OVRInput.Controller.LTouch)
            channel = OVRHaptics.LeftChannel;

        for (float t = 0; t <= time; t += Time.deltaTime)
        {
            switch (force)
            {
                case VibrationForce.Light:
                    channel.Queue(clipLight);
                    break;
                case VibrationForce.Medium:
                    channel.Queue(clipMedium);
                    break;
                case VibrationForce.Hard:
                    channel.Queue(clipHard);
                    break;
            }
        }
        yield return new WaitForSeconds(time);
        channel.Clear();
        //forcedHaptic = false;
        yield return null;

    }
}
