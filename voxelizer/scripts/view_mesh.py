import os
import numpy as np
from glumpy import app, gl, gloo, data, log
from glumpy.transforms import Trackball, Position

vertex = """
uniform mat4 m_model;
uniform mat4 m_view;
uniform mat4 m_normal;
attribute vec3 position;
attribute vec3 normal;
varying vec3 v_normal;
varying vec3 v_position;
void main()
{
    gl_Position = <transform>;
    vec4 P = m_view * m_model* vec4(position, 1.0);
    v_position = P.xyz / P.w;
    v_normal = vec3(m_normal * vec4(normal,0.0));
}
"""

fragment = """
varying vec3 v_normal;
varying vec3 v_position;
const vec3 light_position = vec3(1.0,1.0,1.0);
const vec3 ambient_color = vec3(0.1, 0.0, 0.0);
const vec3 diffuse_color = vec3(0.75, 0.125, 0.125);
const vec3 specular_color = vec3(1.0, 1.0, 1.0);
const float shininess = 128.0;
const float gamma = 2.2;
void main()
{
    vec3 normal= normalize(v_normal);
    vec3 light_direction = normalize(light_position - v_position);
    float lambertian = max(dot(light_direction,normal), 0.0);
    float specular = 0.0;
    if (lambertian > 0.0)
    {
        vec3 view_direction = normalize(-v_position);
        vec3 half_direction = normalize(light_direction + view_direction);
        float specular_angle = max(dot(half_direction, normal), 0.0);
        specular = pow(specular_angle, shininess);
    }
    vec3 color_linear = ambient_color +
                        lambertian * diffuse_color +
                        specular * specular_color;
    vec3 color_gamma = pow(color_linear, vec3(1.0/gamma));
    gl_FragColor = vec4(color_gamma, 0.2);
}
"""

log.info("Loading cubes mesh")

script_dir = os.path.dirname(__file__)
os.chdir(script_dir)

mesh_file = os.path.join(script_dir,'teste.obj')
vertices,indices = data.get(mesh_file)

mesh = gloo.Program(vertex, fragment)
mesh.bind(vertices)
trackball = Trackball(Position("position"))
mesh['transform'] = trackball
trackball.theta, trackball.phi, trackball.zoom = 80, -135, 5

window = app.Window(width=1024, height=768)

def update():
    model = mesh['transform']['model'].reshape(4,4)
    view  = mesh['transform']['view'].reshape(4,4)
    mesh['m_view']  = view
    mesh['m_model'] = model
    mesh['m_normal'] = np.array(np.matrix(np.dot(view, model)).I.T)
    
@window.event
def on_draw(dt):
    window.clear()
    mesh.draw(gl.GL_TRIANGLES)

@window.event
def on_mouse_drag(x, y, dx, dy, button):
    update()
    
@window.event
def on_init():
    gl.glEnable(gl.GL_DEPTH_TEST)
    update()

window.attach(mesh['transform'])
app.run()