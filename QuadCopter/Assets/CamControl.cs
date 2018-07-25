using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CamControl : MonoBehaviour {
    Rigidbody body;
    public Vector3 spawn;
	// Use this for initialization
	void Start () {
        body = GetComponent<Rigidbody>();
        spawn = transform.position;
	}
    public void r()
    {
        body.velocity = new Vector3(0f, 0f, 0f);
        body.angularVelocity = new Vector3(0f, 0f, 0f);
        body.rotation = Quaternion.Euler(new Vector3(0f, 0f, 0f));
        body.position = spawn;
        transform.rotation = Quaternion.Euler(new Vector3(0f, 0f, 0f));
        transform.position = spawn;
        body.Sleep();
    }
    // Update is called once per frame
    void Update () {
		
	}
}
