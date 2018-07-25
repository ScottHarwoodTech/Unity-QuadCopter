using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class thrusterControl : MonoBehaviour {
    public Rigidbody body;
    public int id = 0;
    public Control target;
    Vector3 spawn;
    FixedJoint joint;
    public ParticleSystem system;
    bool kinematic = false;
	// Use this for initialization
	void Start () {
        joint = GetComponent<FixedJoint>();
        body = GetComponent<Rigidbody>();
        system = GetComponent<ParticleSystem>();
        spawn = transform.position;
	}

    public void r(Control target)
    {
        body.velocity = new Vector3(0f, 0f, 0f);
        body.angularVelocity = new Vector3(0f, 0f, 0f);
        body.rotation = Quaternion.Euler(new Vector3(0f, 0f, 0f));
        body.position = spawn;
        transform.rotation = Quaternion.Euler(new Vector3(0f, 0f, 0f));
        transform.position = spawn;
        body.Sleep();
        // kinematic = true;
    }
    public void linkedTick()
    {
       if (kinematic)
        {
            body.isKinematic = true;
            kinematic = false;
        }
        else
        {
            body.isKinematic = false;
        }
    }
}
