using System.Collections;
using System.Collections.Generic;

using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;
public class Control : MonoBehaviour
{
    public float maxThrust = 100;
    public int maxPart = 2;
    public Vector3 localrotate;
    public Vector3 localVelocity;
    public int scale = 100;
    public bool crashed = false;
    //public Debugger Debug;
    private Vector3 spawn;
    public float[] thrusts = new float[4];//a list for each prop thrust
    public thrusterControl[] thrusters = new thrusterControl[4];
    public Rigidbody body;
    public bool resetBool = false;
    public SocketHandler controller;
    public CamControl cam;
    bool kinematic = false;
    public Text target;
    // Use this for initialization
    void Start()
    {

        localrotate = transform.localEulerAngles;
        Debug.Log("HELLO WORLD");
        body = GetComponent<Rigidbody>();
        localVelocity = body.velocity;
        spawn = transform.position;
        //setThrusts("1,1,1,1");
    }

    public void reset()
    {
        
        //controller.control = Instantiate(this, spawn, Quaternion.Euler(new Vector3(0f, 0f, 0f)));
        for (int i = 0; i < thrusts.Length; i++)
        {
            thrusters[i].r(this);
        }
        cam.r();
        /*
        controller.control.thrusters = thrusters;
        cam.j.connectedBody = controller.control.body;
        controller.control.cam = cam;
        controller.control.cam.transform.position = controller.control.cam.spawn;
        Destroy(gameObject);
        */
       
        body.velocity = new Vector3(0f, 0f, 0f);
        body.angularVelocity = new Vector3(0f, 0f, 0f);
        body.rotation = Quaternion.Euler(new Vector3(0f, 0f, 0f));
        //body.MoveRotation(Quaternion.Euler(new Vector3(0f, 0f, 0f)));
        //body.MovePosition(spawn);
        body.position = spawn;
        transform.rotation = Quaternion.Euler(new Vector3(0f, 0f, 0f));
        transform.position = spawn;
        setThrusts("0,0,0,0");
        kinematic = true;
    }
    public void setThrusts(string data)
    {
        string[] forces = data.Split(',');
        for (int i = 0; i < thrusts.Length; i++)
        {
            thrusts[i] = float.Parse(forces[i]);//set each thrust
        }
    }
    // Update is called once per frame
    void Update()
    {
        target.text = thrusts[0].ToString() + " " + thrusts[1].ToString() + " " + thrusts[2].ToString() + " " + thrusts[3].ToString();
        for (int i = 0; i < thrusts.Length; i++)
        {
            thrusters[i].body.AddRelativeForce(new Vector3(0, thrusts[i] * maxThrust, 0));
            var m = thrusters[i].system.main;
            m.simulationSpeed = thrusts[i] * maxPart;
        }
        localrotate = transform.localEulerAngles;
        localVelocity = body.velocity;
        //detect crashing
        if (resetBool)
        {
            reset();
            resetBool = false;//reset the boolean
        }
        if (kinematic)
        {
            body.isKinematic = true;
            kinematic = false;
            for (int i = 0; i < thrusts.Length; i++)
            {
                thrusters[i].linkedTick();
            }
        }
        else
        {
            body.isKinematic = false;
            for (int i = 0; i < thrusts.Length; i++)
            {
                thrusters[i].linkedTick();
            }
        }
    }
    private void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.name == "Terrain")
        {
            crashed = true;
            reset();
        }
    }
}

