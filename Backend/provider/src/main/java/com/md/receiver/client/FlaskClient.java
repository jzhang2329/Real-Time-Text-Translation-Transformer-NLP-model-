package com.md.receiver.client;

import org.springframework.cloud.openfeign.FeignClient;
import org.springframework.stereotype.Component;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;

@Component
@FeignClient(name = "flaskClient", url = "http://127.0.0.1:8000")
//@RequestMapping("/api")
public interface FlaskClient {

    @GetMapping("/result")
    String service();

//    @PostMapping("/data")
//    String postData();

}
