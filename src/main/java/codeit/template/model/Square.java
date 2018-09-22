package codeit.template.model;

import com.fasterxml.jackson.annotation.JsonProperty;

public class Square {
    @JsonProperty("input")
    private final Integer input;

    public Square(@JsonProperty("input") Integer input){
        this.input = input;
    }


    public Integer getInput() {
        return input;
    }
}
