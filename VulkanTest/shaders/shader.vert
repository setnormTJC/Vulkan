#version 450


layout(location = 0) in vec2 inPosition; //32 bit floats
layout(location = 1) in vec3 inColor; //loc is an "attribute index" 

layout(location = 0) out vec3 fragColor;


void main() {
	gl_Position = vec4(inPosition, 0.0, 1.0); 
	fragColor = inColor; //a vec3 - interpolates color between the three points
		//fragColor gets sent to frag shader as an input
}