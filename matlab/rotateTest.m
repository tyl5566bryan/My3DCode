function F = rotateTest(F, theta)

vertices = F.vertices;

R = [cos(theta), sin(theta), 0; -sin(theta), cos(theta), 0; 0, 0, 1];

vertices = vertices * R;

F.vertices = vertices;

end