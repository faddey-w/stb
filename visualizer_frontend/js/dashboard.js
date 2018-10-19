
angular.module("StrateobotsApp", [
    'ngStorage'
]).controller("StrateobotsController", function($scope, $timeout, $localStorage, $http, $q) {

    $scope.simulations = $localStorage.$default({
        'simulations': []
    }).simulations;
    $scope.status = {
        queue_size: '?',
        max_queue: '?',
        currently_runs: false,
        isCurrentlyLoading: function() {
            return !!$scope.currentSim && $scope.currentSim.n_loaded < $scope.currentSim.ticks_generated;
        },
        isAnimRunning: function () {
            return !!animator && animator.isRunning();
        },
        getCurrentFrame: function () {
            if (!!animator) return animator.getCurrentFrame()
        },
        getFramesCount: function() {
            if (!!renderer) return renderer.getFramesCount();
        }
    };
    $scope.online = false;
    $scope.ping = 999;
    $scope.addSimulation = addSimulation;

    var renderer = new Renderer(document.getElementById('render_field'));
    var animator = null;

    $scope.currentSim = null;
    $scope.animProgress = 0;
    $scope.setCurrentSimulation = function(sim) {
        if (!!$scope.currentSim) $scope.currentSim.n_loaded = 0;
        sim.n_loaded = 0;
        $scope.currentSim = sim;
        loadSimulationData(sim, 300);
    };
    $scope.toggleAnimation = function () {
        if (animator.isRunning()) {
            animator.stop();
        } else {
            animator.start();
        }
    };
    $scope.goToFrameRel = function (rel) {
        animator.skipToFrame(animator.getCurrentFrame() + rel);
        animator.update();
    };
    $scope.setAnimProgress = function(value) {
        animator.skipToPercent(value);
    };
    $scope.cleanup = function() {
        var cleanedSimulations = $scope.simulations.filter(function (sim) { return !sim.has_error; });
        $scope.simulations.splice(0, $scope.simulations.length);
        cleanedSimulations.forEach(function (sim) {
            $scope.simulations.push(sim)
        });
    };
    $scope.onProgressClick = function (e) {
        var ratio = e.offsetX / e.currentTarget.clientWidth;
        animator.skipToPercent(ratio);
        animator.update();
    };

    loopLoadUpdate();

    function loadUpdate() {
        var promises = [
            $http.get('/api/v1/simulation').then(function (data) {
                data = data.data;
                $scope.status.queue_size = data.queue_size;
                $scope.status.max_queue = data.max_queue;
                $scope.status.currently_runs = data.currently_runs;
                $scope.online = true;
            }, onError)
        ];
        $scope.simulations.forEach(function(sim) {
            if (sim.has_error) return;
            promises.push($http.get('/api/v1/simulation/'+sim.id).then(function(response) {
                if (response.status == 404) {
                    sim.has_error = true;
                    sim.error = response.statusText;
                    return;
                }
                var data = response.data;
                sim.started = data.started;
                sim.ticks_generated = data.ticks_generated;
                sim.finished = data.finished;
                sim.cancelled = data.cancelled;
                $scope.online = true
            }, function (resp) {
                sim.has_error = true;
                sim.error = resp.statusText;
                onError();
            }));
        });
        return promises;

        function onError() {
            $scope.online = false;
        }
    }
    function loopLoadUpdate() {
        var started_at = Date.now();
        $q.all(loadUpdate()).then(_schedule, _schedule);

        function _schedule() {
            var done_at = Date.now();
            $scope.ping = done_at-started_at;
            $timeout(loopLoadUpdate, Math.max(0, 1000-(done_at-started_at)));
        }
    }

    function addSimulation() {
        $http({
            method: 'POST',
            url: '/api/v1/simulation',
            data: 'width=1000&height=1000',
            headers: {'Content-Type': 'application/x-www-form-urlencoded;charset=utf-8'}
        }).then(function(data) {
            data = data.data;
            $scope.simulations.push({id: data.id, has_error: false, data: null});
        })
    }

    loopUpdateAnimProgress();
    function loopUpdateAnimProgress() {
        if (animator !== null) {
            $scope.animProgress = animator.getCurrentFrame() / renderer.getFramesCount();
        }
        $timeout(loopUpdateAnimProgress, 500);
    }

    function loadSimulationData(sim, chunkSize) {
        var loadedFrames = [];
        function loadChunk(start) {
            return $http.get('/api/v1/simulation/'+sim.id+'/data?start='+start+'&count='+chunkSize)
                .then(function(response) {
                    loadedFrames = loadedFrames.concat(response.data.data);
                    if (start == 0 && loadedFrames.length > 0) {
                        renderer.clearDataFrames();
                        renderer.addDataFrame(
                            loadedFrames[0].bots,
                            loadedFrames[0].bullets,
                            loadedFrames[0].rays,
                            loadedFrames[0].explosions
                        );
                        renderer.renderFrame(0);
                    }
                    start += response.data.count;
                    sim.n_loaded += response.data.count;
                    if (start < sim.ticks_generated) return loadChunk(start);
                });
        }
        return loadChunk(0).then(function() {
            renderer.clearDataFrames();
            for(var i = 0; i < sim.ticks_generated; i++) {
                renderer.addDataFrame(
                    loadedFrames[i].bots,
                    loadedFrames[i].bullets,
                    loadedFrames[i].rays,
                    loadedFrames[i].explosions
                );
            }
            animator = new Animator(renderer, 50);
        })
    }

});

