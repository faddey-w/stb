
function Renderer(container) {
    var HEAVY = 1, RAIDER = 2, SNIPER = 3;

    var camera, scene, renderer;
    var worldSize = 1000;
    var cameraXOffs = 0;
    var cameraYOffs = 0;
    var cameraScale = 1;

    // data of battle movie
    var _frames = [];

    // cache of ThreeJS objects, reused every time when we render a frame
    var _tanks = {keys: []};
    var _bullets = {}, _rays = {};
    var _explosions = [];
    var _aims = [];
    [HEAVY, RAIDER, SNIPER].forEach(function(t) {
        _bullets[t] = [];
        _rays[t] = [];
    });

    var params = {
        show_aims: false,
    };

    this.addDataFrame = function(bots, bullets, rays, explosions, aims) {
        _frames.push({
            bots: bots,
            bullets: bullets,
            rays: rays,
            explosions: explosions,
            aims: aims || [],
        });
    };
    this.clearDataFrames = function() {
        _frames = [];
    };
    this.renderFrame = function(frameId) {
        var bots = _frames[frameId].bots,
            bullets = _frames[frameId].bullets,
            rays = _frames[frameId].rays,
            explosions = _frames[frameId].explosions,
            aims = _frames[frameId].aims;
        var botCounters = {},
            i,
            bulletCounters = {},
            rayCounters = {},
            explosionCounter = 0,
            aimsCounter = 0;

        for (i = 0; i < bots.length; i++) {
            var bot = bots[i];
            var key = [bot.type, bot.team];
            var collection;
            if (key in _tanks) {
                collection = _tanks[key];
            } else {
                collection = {
                    bots: [],
                    team: bot.team,
                    type: bot.type
                };
                _tanks[key] = collection;
                _tanks.keys.push(key);
            }
            botCounters[key] = (botCounters[key] || 0) + 1;

            var botObj;
            if (botCounters[key] > collection.bots.length) {
                var color = bot.team;
                switch (bot.type) {
                    case HEAVY: botObj = new HeavyTank(color); break;
                    case RAIDER: botObj = new RaiderTank(color); break;
                    case SNIPER: botObj = new SniperTank(color); break;
                    default: throw "unknown type of bot: "+bot.type;
                }
                collection.bots.push(botObj);
                scene.add(botObj.getThreeJsObject());
            } else {
                botObj = collection.bots[botCounters[key]-1];
            }
            botObj.setPosition(bot.x, bot.y);
            botObj.setHP(bot.hp);
            botObj.setCD(bot.load);
            botObj.setOrientation(bot.orientation);
            botObj.setTowerOrientation(bot.tower_orientation);
            botObj.setShieldEnergy(bot.shield);
            botObj.setShieldWarmup(bot.shield_warmup);
            botObj.getThreeJsObject().visible = true;
            if (bot.has_shield) {
                botObj.shield.position.z = 200;
//                botObj.shieldWarmup.position.z = 200;
            } else {
                botObj.shield.position.z = camera.position.z + 100;
//                botObj.shieldWarmup.position.z = camera.position.z + 100;
            }

        }
        bullets.forEach(function(bullet) {
            var key = bullet.type;
            var bulletlist = _bullets[key];
            bulletCounters[key] = (bulletCounters[key] || 0) + 1;
            var bulletObj;
            if (bulletCounters[key] > bulletlist.length) {
                switch (bullet.type) {
                    case HEAVY: bulletObj = new Bullet(35); break;
                    case RAIDER: bulletObj = new Bullet(20); break;
                    case SNIPER: bulletObj = new Bullet(10); break;
                    default: throw "unknown type of bullet: "+bot.type;
                }
                bulletlist.push(bulletObj);
                scene.add(bulletObj.getThreeJsObject());
            } else {
                bulletObj = bulletlist[bulletCounters[key]-1];
            }
            bulletObj.setOrientation(bullet.orientation);
            bulletObj.setPosition(bullet.x, bullet.y);
            bulletObj.getThreeJsObject().visible = true;
        });
        rays.forEach(function(ray) {
            var key = ray.type;
            var raylist = _rays[key];
            rayCounters[key] = (rayCounters[key] || 0) + 1;
            var rayObj;
            if (rayCounters[key] > raylist.length) {
                switch (ray.type) {
                    case HEAVY: rayObj = new Ray(35); break;
                    case RAIDER: rayObj = new Ray(25); break;
                    case SNIPER: rayObj = new Ray(20); break;
                    default: throw "unknown type of ray: "+bot.type;
                }
                raylist.push(rayObj);
                scene.add(rayObj.getThreeJsObject());
            } else {
                rayObj = raylist[rayCounters[key]-1];
            }
            rayObj.setPosition(ray.x, ray.y, ray.orientation, ray.range);
            rayObj.getThreeJsObject().visible = true;
        });
        explosions.forEach(function(explosion) {
            var expl;
            if (explosionCounter >= _explosions.length) {
                expl = new Explosion();
                _explosions.push(expl);
                scene.add(expl.getThreeJsObject());
            } else {
                expl = _explosions[explosionCounter];
            }
            explosionCounter++;
            expl.setPosition(explosion.x, explosion.y);
            expl.setState(explosion.size, explosion.t / explosion.duration);
            expl.getThreeJsObject().visible = true;
        });
        if (params.show_aims) {
            aims.forEach(function(aim) {
                var aimObj;
                if (aimsCounter < _aims.length) {
                    aimObj = _aims[aimsCounter];
                } else {
                    aimObj = new Aim();
                    _aims.push(aimObj);
                    scene.add(aimObj.object);
                }
                aimsCounter++;
                aimObj.object.visible = true;
                aimObj.update(aim.x, aim.y, aim.color);
            });
        }

        // hide everything we don't need
        _tanks.keys.forEach(function(key) {
            var start = botCounters[key] || 0;
            var bots = _tanks[key].bots;
            var cnt = bots.length;
            for(var i = start; i < cnt; i++) {
                bots[i].getThreeJsObject().visible = false;
            }
        });
        [HEAVY, RAIDER, SNIPER].forEach(function(t) {
            for(i = bulletCounters[t] || 0; i < _bullets[t].length; i++) {
                _bullets[t][i].getThreeJsObject().visible = false;
            }
            for(i = rayCounters[t] || 0; i < _rays[t].length; i++) {
                _rays[t][i].getThreeJsObject().visible = false;
            }
        });
        for (i = explosionCounter; i < _explosions.length; i++) {
            _explosions[i].getThreeJsObject().visible = false;
        }
        for (i = aimsCounter; i < _aims.length; i++) {
            _aims[i].object.visible = false;
        }

        // do render
//        camera.lookAt(scene.position);
        renderer.render(scene, camera);
    };
    this.getFramesCount = function() {
        return _frames.length;
    };
    this.showAims = function(value) {
        params.show_aims = value;
    }

    init();
    function init() {
        var aspect = container.innerWidth / container.innerHeight;
        camera = new THREE.OrthographicCamera(worldSize * aspect / -2, worldSize * aspect / 2, worldSize / 2, worldSize / -2, 1, 2000);
        camera.position.z = 1000;

        camera.left = 0;
        camera.right = worldSize;
        camera.bottom = 0;
        camera.top = worldSize;

        scene = new THREE.Scene();

//        camera.lookAt(new THREE.Vector3(
//            scene.position.x,//+cameraXOffs,
//            scene.position.y,//+cameraYOffs,
//            scene.position.z));
        camera.lookAt(scene.position);
        camera.updateProjectionMatrix();

        scene.position.x += cameraXOffs;
        scene.position.y += cameraYOffs;
        scene.scale.x = scene.scale.y = cameraScale;
        scene.background = new THREE.Color(0xffffff);

        // make battle field grey and draw a border around.
        var battlefield = new THREE.Mesh(
            new THREE.PlaneBufferGeometry(worldSize, worldSize),
            new THREE.MeshBasicMaterial({color: new THREE.Color(0xf0f0f0)})
        );
        var border_geometry = new THREE.BufferGeometry();
        border_geometry.setFromPoints([
            {x: 0, y: 0},
            {x: worldSize, y: 0},
            {x: worldSize, y: worldSize},
            {x: 0, y: worldSize}
        ]);
        var battlefield_border = new THREE.LineLoop(
            border_geometry,
            new THREE.LineBasicMaterial({color: new THREE.Color(0x000000)})
        );
        battlefield.position.z = -1000;
        battlefield_border.position.z = -1000;
//        scene.add(battlefield);
        scene.add(battlefield_border);

        renderer = new THREE.WebGLRenderer({antialias: true});
        renderer.setPixelRatio(window.devicePixelRatio);
        // renderer.setSize(window.innerWidth, window.innerHeight);
        container.appendChild(renderer.domElement);
        //
        window.addEventListener('resize', onWindowResize, false);
        window.addEventListener('load', onWindowResize, false);
        onWindowResize();
    }

    function onWindowResize() {

//        camera.position.set(
//            camera.position.x+cameraXOffs,
//            camera.position.y+cameraYOffs,
//            camera.position.z);

        var canvasSize = Math.min(container.clientWidth, container.clientHeight);
        renderer.setSize(canvasSize, canvasSize);

        renderer.render(scene, camera);
    }
}

function Animator(renderer, fps) {
    var _nframes = renderer.getFramesCount();
    var _startTime = Date.now();
    var _frameToRender = 0;
    var _lastRenderedFrame = null;
    var _running = false;
    var _animating = false;

    this.skipToTime = function (sec) {
        _startTime = Date.now() - 1000*sec;
        _frameToRender = calcFrameByTime();
    };
    this.skipToPercent = function(fraction_0to1) {
        var totalsecs = _nframes / fps;
        this.skipToTime(fraction_0to1 * totalsecs);
    };
    this.skipToFrame = function (frameno) {
        frameno = Math.max(0, frameno);
        frameno = Math.min(_nframes, frameno);
        this.skipToTime(frameno / fps);
    };
    this.start = function () {
        _running = true;
        this.skipToFrame(_frameToRender);
        if (!_animating) animateOnce();
    };
    this.stop = function () {
        _running = false;
    };
    this.isRunning = function () {
        return _running;
    };
    this.getCurrentFrame = function() {
        return _frameToRender;
    };
    this.update = update;

    function animateOnce() {
        if (_running) {
            _animating = true;
            _frameToRender = calcFrameByTime();
            update();
            if (_frameToRender >= _nframes) {
                _running = false;
            }
            if (_running) {
                requestAnimationFrame(animateOnce);
            } else {
                _animating = false;
            }
        } else {
            _animating = false;
        }
    }
    function update() {
        if (_frameToRender == _lastRenderedFrame || _frameToRender >= _nframes) return;
        renderer.renderFrame(_frameToRender);
    }
    function calcFrameByTime() {
        var now = Date.now();
        return ((now - _startTime) * fps / 1000) | 0;
    }
}
var axisZ = new THREE.Vector3(0, 0, 1);

function Tank(color) {
    if (this.prototype === Tank) {
        throw Error("Tank is abstract class");
    }
    if (!this._geometry.initialized) {
        this._geometry.body = new THREE.ShapeBufferGeometry(
            new THREE.Shape(this._edgePoints)
        );
        this._geometry.edge.setFromPoints(this._edgePoints);
        this._geometry.initialized = true;
    }

    var body = new THREE.Mesh(this._geometry.body, this._material.body);

    var edge = new THREE.LineLoop(this._geometry.edge, this._material.edge);
    edge.position.z = 10;

    if (this._material.markers[color] === undefined) {
        this._material.markers[color] = new THREE.LineBasicMaterial({color: color})
    }
    var colorMarker = new THREE.LineLoop(this._geometry.edge, this._material.markers[color]);
    colorMarker.scale.x = 0.7;
    colorMarker.scale.y = 0.7;
    colorMarker.position.z = 20;

    var towerBodyEdge = new THREE.Mesh(this._geometry.tower, this._material.edge);
    towerBodyEdge.position.z = 100;

    var towerColorMarker = new THREE.Mesh(this._geometry.tower, this._material.markers[color]);
    towerColorMarker.position.z = 110;
    towerColorMarker.scale.x = 0.85;
    towerColorMarker.scale.y = 0.85;

    var towerBody = new THREE.Mesh(this._geometry.tower, this._material.body);
    towerBody.position.z = 120;
    towerBody.scale.x = 0.6;
    towerBody.scale.y = 0.6;

    var gunEdge = new THREE.Mesh(this._geometry.gun, this._material.edge);
    gunEdge.position.y = this._parameters.gunEdgeY;
    gunEdge.position.z = 50;

    var gunBody = new THREE.Mesh(this._geometry.gun, this._material.body);
    gunBody.position.y = this._parameters.gunBodyY;
    gunBody.position.z = 55;
    gunBody.scale.y = 12.0 / 14.0;
    gunBody.scale.x = 0.75;

    var tower = new THREE.Group();
    tower.add(towerBodyEdge);
    tower.add(towerColorMarker);
    tower.add(towerBody);
//        tower.add(gunBody);
    tower.add(gunEdge);

    var tank = new THREE.Group();
    tank.add(body);
    tank.add(edge);
    tank.add(colorMarker);
    tank.add(tower);

    var shieldBar = new THREE.Mesh(this._healthBarGeometry, this._material.shieldBar);
    shieldBar.position.y = 205;
    shieldBar.position.z = 101;
    var shieldBarBackground = new THREE.Mesh(this._healthBarGeometry, this._material.barBackground);
    shieldBarBackground.position.y = 205;
    shieldBarBackground.position.z = 100;

    var health = new THREE.Mesh(this._healthBarGeometry, this._material.health);
    health.position.y = 190;
    health.position.z = 101;
    var healthBarBackground = new THREE.Mesh(this._healthBarGeometry, this._material.barBackground);
    healthBarBackground.position.y = 190;
    healthBarBackground.position.z = 100;

    var cooldown = new THREE.Mesh(this._healthBarGeometry, this._material.cooldown);
    cooldown.position.y = 175;
    cooldown.position.z = 101;
    var cooldownBarBackground = new THREE.Mesh(this._healthBarGeometry, this._material.barBackground);
    cooldownBarBackground.position.y = 175;
    cooldownBarBackground.position.z = 100;

    var shield = new THREE.Mesh(this._shieldGeometry, this._material.shield);
    shield.position.z = 200;
    var shieldWarmup = new THREE.Mesh(this._shieldEdgeGeometry, this._material.shieldWarmup);
    shieldWarmup.position.z = 200;

    var unit = new THREE.Group();
    unit.add(tank);
    unit.add(healthBarBackground);
    unit.add(cooldownBarBackground);
    unit.add(shieldBarBackground);
    unit.add(shieldBar);
    unit.add(health);
    unit.add(cooldown);
    unit.add(shield);
    unit.add(shieldWarmup);
    unit.scale.x = 0.2;
    unit.scale.y = 0.2;

    this.unit = unit;
    this.health = health;
    this.cooldown = cooldown;
    this.shieldBar = shieldBar;
    this.machine = tank;
    this.tower = tower;
    this.shield = shield;
    this.shieldWarmup = shieldWarmup;
}

Tank.prototype = {
    constructor: Tank,
    _material: {
        body: new THREE.MeshBasicMaterial({color: 0x909090}),
        edge: new THREE.LineBasicMaterial({color: 0x000000}),
        health: new THREE.MeshBasicMaterial({color: 0xEF1010}),
        cooldown: new THREE.MeshBasicMaterial({color: 0xDFDF44}),
        shieldBar: new THREE.MeshBasicMaterial({color: 0x42c5f4}),
        barBackground: new THREE.MeshBasicMaterial({color: 0x202020}),
        shield: new THREE.MeshBasicMaterial({color: 0x42c5f4, opacity: 0.5, transparent: true}),
        shieldWarmup: new THREE.MeshBasicMaterial({color: 0x42c5f4, opacity: 0.7, transparent: true}),
        markers: {}
    },
    _edgePoints: [
        {x: -90, y: -130},
        {x: -110, y: -110},
        {x: -110, y: +110},
        {x: -90, y: +130},
        {x: +90, y: +130},
        {x: +110, y: +110},
        {x: +110, y: -110},
        {x: +90, y: -130}
    ],
    _healthBarGeometry: new THREE.PlaneBufferGeometry(250, 15),
    _shieldGeometry: new THREE.CircleBufferGeometry(150, 32),
    _shieldEdgeGeometry: new THREE.RingBufferGeometry(140, 150, 32),
    _geometry: {
        initialized: false,
        edge: new THREE.BufferGeometry(),
        tower: new THREE.CircleBufferGeometry(60, 16),
        gun: new THREE.PlaneBufferGeometry(40, 140),
        body: new THREE.Geometry()
    },
    _parameters: {
        gunEdgeY: 0,
        gunBodyY: 0
    },
    setOrientation: function (angle) {
        this.machine.setRotationFromAxisAngle(axisZ, (angle - Math.PI / 2));
    },
    setTowerOrientation: function (angle) {
        this.tower.setRotationFromAxisAngle(axisZ, angle);
    },
    setPosition: function (x, y) {
        this.unit.position.x = x;
        this.unit.position.y = y;
    },
    setHP: function (value0to1) {
        var totalWidth = 250; // this._healthBarGeometry.width;
        this.health.scale.x = value0to1;
        this.health.position.x = totalWidth * (value0to1 - 1) / 2;
    },
    setCD: function (value0to1) {
        var totalWidth = 250; // this._healthBarGeometry.width;
        this.cooldown.scale.x = value0to1;
        this.cooldown.position.x = totalWidth * (value0to1 - 1) / 2;
    },
    setShieldWarmup: function(value0to1) {
        var scale = value0to1 > 0.01 ? 0.2 + value0to1 * 0.8 : 0;
        this.shieldWarmup.scale.x = scale;
        this.shieldWarmup.scale.y = scale;
    },
    setShieldEnergy: function(value0to1) {
        var totalWidth = 250; // this._healthBarGeometry.width;
        this.shieldBar.scale.x = value0to1;
        this.shieldBar.position.x = totalWidth * (value0to1 - 1) / 2;
    },
    getThreeJsObject: function () {
        return this.unit;
    }
};

function HeavyTank(color) {
    Tank.call(this, color);
}

HeavyTank.prototype = Object.assign(Object.create(Tank.prototype), {
    constructor: HeavyTank,
    _edgePoints: [
        {x: -90, y: -130},
        {x: -110, y: -110},
        {x: -110, y: +110},
        {x: -90, y: +130},
        {x: +90, y: +130},
        {x: +110, y: +110},
        {x: +110, y: -110},
        {x: +90, y: -130}
    ],
    _parameters: {
        gunEdgeY: 80,
        gunBodyY: 80
    },
    _geometry: {
        initialized: false,
        edge: new THREE.BufferGeometry(),
        tower: new THREE.CircleBufferGeometry(60, 16),
        gun: new THREE.PlaneBufferGeometry(40, 140),
        body: new THREE.Geometry()
    }
});

function RaiderTank(color) {
    Tank.call(this, color);
}

RaiderTank.prototype = Object.assign(Object.create(Tank.prototype), {
    constructor: RaiderTank,
    _edgePoints: [
        {x: -80, y: -100},
        {x: -90, y: -80},
        {x: -90, y: +40},
        {x: -60, y: +100},
        {x: +60, y: +100},
        {x: +90, y: +40},
        {x: +90, y: -80},
        {x: +80, y: -100}
    ],
    _parameters: {
        gunEdgeY: 70,
        gunBodyY: 70
    },
    _geometry: {
        initialized: false,
        edge: new THREE.BufferGeometry(),
        tower: new THREE.PlaneBufferGeometry(70, 70),
        gun: new THREE.PlaneBufferGeometry(25, 80),
        body: new THREE.Geometry()
    }
});

function SniperTank(color) {
    Tank.call(this, color);
    this.tower.position.y = -60;
}

SniperTank.prototype = Object.assign(Object.create(Tank.prototype), {
    constructor: SniperTank,
    _edgePoints: [
        {x: -90, y: -130},
        {x: -110, y: -110},
        {x: -110, y: -50},
        {x: -20, y: +160},
        {x: +20, y: +160},
        {x: +110, y: -50},
        {x: +110, y: -110},
        {x: +90, y: -130}
    ],
    _parameters: {
        gunEdgeY: 140,
        gunBodyY: 140
    },
    _geometry: {
        edge: new THREE.BufferGeometry(),
        tower: new THREE.CircleBufferGeometry(50, 6),
        gun: new THREE.PlaneBufferGeometry(20, 220),
        body: new THREE.Geometry()
    }
});

function Bullet(size) {
    var object = new THREE.Mesh(this._geometry, this._material);
    object.scale.x = size / 10.0;
    object.scale.y = size / 10.0;
    object.position.z = 900;
    this.object = object;
}

Bullet.prototype = {
    constructor: Bullet,
    _geometry: new THREE.PlaneBufferGeometry(1, 3),
    _material: new THREE.MeshBasicMaterial({color: 0xFF0000}),
    setOrientation: function (angle) {
        var obj = this.getThreeJsObject();
        obj.setRotationFromAxisAngle(axisZ, (angle - Math.PI / 2));
    },
    setPosition: function (x, y) {
        var obj = this.getThreeJsObject();
        obj.position.x = x;
        obj.position.y = y;
    },
    getThreeJsObject: function () {
        return this.object;
    }
};

function Ray(width) {
    this.object = new THREE.Group();
    this.object.add(new THREE.Mesh(this._geometry, this._material.ray));
//        this.flashes = [];
//        this.flashPositions = [];
//        for(var i = 0; i < 10; i++
//            this.flashes.push(flash);
//            this.flashPositions.push(Math.random());) {
//            var flash = new THREE.Mesh(this._geometry, this._material.flash);
//            flash.position.x = Math.random();
//            this.object.add(flash);
//        }
    this.object.scale.y = width * 0.1;
    this.object.position.z = 900;

}

Ray.prototype = {
    constructor: Ray,
    _material: {
        ray: new THREE.MeshBasicMaterial({color: 0x2d8bff}),
        flash: new THREE.MeshBasicMaterial({color: 0xFFFFFF})
    },
    _geometry: new THREE.PlaneBufferGeometry(1, 1),
    getThreeJsObject: function () {
        return this.object;
    },
    setPosition: function (fromX, fromY, angle, length) {
        var angleRad = angle;
        var toX = length * Math.cos(angleRad) + fromX;
        var toY = length * Math.sin(angleRad) + fromY;
        this.object.position.x = (toX + fromX) / 2;
        this.object.position.y = (toY + fromY) / 2;
        this.object.scale.x = length;
        this.object.setRotationFromAxisAngle(axisZ, angleRad);
//            var flashScale = 1.0 / length;
//            for(var i = 0; i < this.flashes.length; i++) {
//                this.flashes[i].scale.x = flashScale;
//            }
    },
    updateAnimation: function () {
//            var shift = Date.now() * 0.01;
//            shift = shift - (shift | 0);
//            for(var i = 0; i < this.flashes.length; i++) {
//                var flashShift = shift + this.flashPositions[i];
//                flashShift = flashShift - (flashShift | 0);
//                this.flashes[i].position.y = flashShift;
//            }
    }
};


function Explosion() {
    this.object = new THREE.Mesh(this._geometry, this._material);
    this.object.z = 1500;
}

Explosion.prototype = {
    constructor: Explosion,
    _geometry: new THREE.CircleBufferGeometry(0.5, 32),
    _material: new THREE.MeshBasicMaterial({color: 0xFF0000}),
    setPosition: function(x, y) {
        this.object.position.x = x;
        this.object.position.y = y;
    },
    setState: function(size, time) {
        var scale;
        if (time > 0.5) {
            // range [0.5-1], values [1-0]
            scale = 2 * (1 - time);
        } else if (time > 0.25) {
            // range [0.25-0.5], values [0.75-1]
            scale = 0.5 + time;
        } else if (time > 0.1) {
            // range [0.1-0.25], values [1-0.75]
            scale = (3.5 - 5 * time) / 3;
        } else {
            // range [0-0.1], values [0-1]
            scale = 10 * time;
        }
        this.object.scale.x = scale * size;
        this.object.scale.y = scale * size;
    },
    getThreeJsObject: function () {
        return this.object;
    }
};


function Aim() {
    this.object = new THREE.Mesh(
        this._edgeGeometry,
        new THREE.MeshBasicMaterial({color: 0x000000})
    );
    this.object.position.z = 950;
    this.object.scale.x = 3;
    this.object.scale.y = 3;
}

Aim.prototype = {
    constructor: Aim,
    _edgeGeometry: new THREE.RingBufferGeometry(3, 4, 8),
    update: function(x, y, color) {
        this.object.position.x = x;
        this.object.position.y = y;
        this.object.material.color.setHex(color);
    }
}
