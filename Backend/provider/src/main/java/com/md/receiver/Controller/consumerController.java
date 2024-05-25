package com.md.receiver.Controller;

import com.md.receiver.client.FlaskClient;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class consumerController {

    @Autowired
    private FlaskClient flaskClient;

    @GetMapping("/test")
    public String service(){
        return flaskClient.service();
    }

}
