
angular.module("StrateobotsApp", [
    'ngStorage'
]).controller("StrateobotsController", function($scope, $timeout, $localStorage, $http, $q) {

    $scope.isAdminEnabled = function() { return true; };
    $scope.simulations = [];
    $scope.botInitializers = [];
    $scope.aiFunctions = [];
    $scope.newGameParams = {
        botInitializerId: null,
        ai1FunctionId: null,
        ai2FunctionId: null,
    };
    $scope.status = {
        is_currently_loading: false,
        isAnimRunning: function () {

            return !!animator && animator.isRunning();
        },
        getCurrentFrame: function () {
            if (!!animator)
                return animator.getCurrentFrame()
        },
        getFramesCount: function() {
            if (!!renderer)
                return renderer.getFramesCount();
        }
    };
    $scope.refresh = function() {
        refreshLaunchParamsList();
        refreshGamesList();
    };
    $scope.startNewGame = function() {
        $http.post('/api/v1/game', {
            initializer_id: $scope.newGameParams.botInitializerId,
            ai1_id: $scope.newGameParams.ai1FunctionId,
            ai2_id: $scope.newGameParams.ai2FunctionId
        }).then(refreshGamesList);
    };

    $scope.currentSim = null;
    $scope.animProgress = 0;
    $scope.setCurrentSimulation = function(sim) {
        if ($scope.status.is_currently_loading) return;
        $scope.currentSim = sim;
        if (sim.n_loaded < sim.nticks) {
            loadSimulationData(sim, 300).then(function() {
                loadGameToRenderer(sim);
            });
        } else {
            loadGameToRenderer(sim);
        }
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
    $scope.removeGame = function(sim) {
        return $http.delete('/api/v1/game/'+sim.id)
                    .finally(refreshGamesList);
    }

    var renderer = new Renderer(document.getElementById('render_field'));
    var animator = null;

//    loopLoadUpdate();
    refreshLaunchParamsList();
    refreshGamesList();

    function refreshLaunchParamsList() {
        return $http.get('/api/v1/launch-params').then(function(resp) {
            $scope.botInitializers = resp.data.bot_initializers;
            $scope.aiFunctions = resp.data.ai_functions;
        });
    }
    function refreshGamesList() {
        var prevGames = $scope.simulations;
        return $http.get('/api/v1/game').then(function(resp) {
            $scope.simulations = resp.data.result;
            var scheduledNext = false;
            $scope.simulations.forEach(function(game) {
                game.frames = [];
                game.n_loaded = 0;
                prevGames.forEach(function(prevGame) {
                    if (game.id == prevGame.id) {
                        game.frames = prevGame.frames;
                        game.n_loaded = prevGame.n_loaded;
                    }
                });

                if (!scheduledNext && !game.finished) {
                    $timeout(refreshGamesList, 1000);
                    scheduledNext = true;
                }
            });
        }, function(error) {
            $timeout(refreshGamesList, 5000);
        });
    }

    loopUpdateAnimProgress();
    function loopUpdateAnimProgress() {
        if (animator !== null) {
            $scope.animProgress = animator.getCurrentFrame() / renderer.getFramesCount();
        }
        $timeout(loopUpdateAnimProgress, 500);
    }

    function loadSimulationData(sim, chunkSize) {
        $scope.status.is_currently_loading = true;
        function loadChunk(start) {
            return $http.get('/api/v1/game/'+sim.id+'?start='+start+'&count='+chunkSize)
                .then(function(response) {
                    sim.frames = sim.frames.concat(response.data.data);
                    sim.n_loaded = sim.frames.length;

                    // show the very first frame on the viewer once it is loaded
                    if (start == 0 && sim.frames.length > 0) {
                        loadGameToRenderer(sim);
                    }

                    start = sim.frames.length;
                    if (start < sim.nticks && sim.frames.length > 0) {
                        return loadChunk(start);
                    }
                });
        }
        return loadChunk(sim.n_loaded).finally(function() {
            $scope.status.is_currently_loading = false;
        })
    }

    function loadGameToRenderer(sim) {
        if (!!animator && animator.isRunning()) {
            animator.stop();
        }
        renderer.clearDataFrames();
        for(var i = 0; i < sim.frames.length; i++) {

            var bots = []
            Object.keys(sim.frames[i].bots).forEach(function(team) {
                var team_int = team | 0;
                sim.frames[i].bots[team].forEach(function(bot) {
                    bot.team = team_int;
                    bots.push(bot);
                });
            });

            renderer.addDataFrame(
                bots,
                sim.frames[i].bullets,
                sim.frames[i].rays,
                sim.frames[i].explosions
            );
        }
        animator = new Animator(renderer, 50);
        renderer.renderFrame(0);
    }

});

