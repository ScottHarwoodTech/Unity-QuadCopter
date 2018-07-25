using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DebugLogger : MonoBehaviour {

    Rigidbody body;
    public Vector3 localrotate;
    public Vector3 localVelocity;
	// Use this for initialization
	void Start () {
        localrotate = transform.localEulerAngles;
        body = GetComponent<Rigidbody>();
        localVelocity = body.velocity;
    }
	
	// Update is called once per frame
	void Update () {
        localVelocity = body.velocity;
        localrotate = transform.localEulerAngles;
    }
}
