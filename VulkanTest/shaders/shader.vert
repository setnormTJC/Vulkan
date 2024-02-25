#version 450

layout(binding = 0) uniform UniformBufferObject 
{
	mat4 model; 
	mat4 view; 
	mat4 proj; 
} ubo; 


layout(location = 0) in vec3 inPosition; //32 bit floats
layout(location = 1) in vec3 inColor; //loc is an "attribute index" 
layout(location = 2) in vec2 inTexCoord; 


layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragTexCoord;


void main() {
	gl_Position = ubo.proj * ubo.view * ubo.model *vec4(inPosition, 1.0); 
	fragColor = inColor; //a vec3 - interpolates color between the three points
							//fragColor gets sent to frag shader as an input
	fragTexCoord = inTexCoord;
}