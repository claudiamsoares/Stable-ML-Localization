clear all;

disp('MWE for C. Soares, J. Xavier and J. Gomes, "Distributed, simple and stable network localization," in Signal and Information Processing (GlobalSIP), 2014 IEEE Global Conference on, pp. 764-768, Dec 2014.')


sensors = [2 -0.8; 3 -0.8; 1 0; 4 0.6; 2.5 0.8]';
anchors = [0, 1; 0, -1; 5, 1; 5, -1]';
distances = dist([anchors sensors]);
distances(distances > 2.2) = 0;
init = sensors + 0.01*randn(size(sensors));
[estimate, nIter] = StableML_Localization(init, anchors, distances);


disp(['Localization error per sensor: '...
      num2str(norm(sensors-estimate,'fro')/size(sensors,2))])
